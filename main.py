import cv2
import re
import sqlite3
import time

from collections import deque
from datetime import datetime
from ultralytics import YOLO


# ============================================================
# OPTIONAL SERIAL SUPPORT
# ============================================================

try:
    import serial
    from serial.tools import list_ports

    SERIAL_AVAILABLE = True

except ImportError:
    serial = None
    list_ports = None
    SERIAL_AVAILABLE = False


# ============================================================
# GENERAL SETTINGS
# ============================================================

MODEL_PATH = "yolo26s.pt"
TRACKER_PATH = "fasttrack.yaml"
DATABASE_PATH = "fair_stand.db"

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

YOLO_IMAGE_SIZE = 320
YOLO_CONFIDENCE = 0.3

PASSERBY_TIME = 1.5
VISITOR_TIME = 5.0
LOST_TIME = 1.5


# ============================================================
# ESP32 SETTINGS
# ============================================================

# Set this to a specific port if automatic detection does not
# select the correct device.
#
# Windows:
# ESP32_PORT = "COM3"
#
# Linux:
# ESP32_PORT = "/dev/ttyUSB0"
#
# macOS:
# ESP32_PORT = "/dev/cu.usbserial-0001"
#
# None enables automatic port detection.
ESP32_PORT = None

ESP32_BAUD_RATE = 115200

ESP32_RECONNECT_INTERVAL = 3.0
ESP32_BOOT_WAIT = 2.0
ESP32_HANDSHAKE_TIMEOUT = 2.0

# Request temperature and humidity every five seconds.
ESP32_TELEMETRY_INTERVAL = 5.0

MAX_DISPLAY_VISITOR_COUNT = 99999
MAX_PENDING_TELEMETRY_SAMPLES = 1000


# ============================================================
# ESP32 CONNECTION
# ============================================================

class ESP32Bridge:
    TELEMETRY_PATTERN = re.compile(
        r'<\*Tele:\["'
        r'(-?\d+(?:\.\d+)?)'
        r'","'
        r'(-?\d+(?:\.\d+)?)'
        r'"\]\*>',
        re.IGNORECASE
    )

    TELEMETRY_ERROR_PATTERN = re.compile(
        r'<\*TeleError:"([^"]+)"\*>',
        re.IGNORECASE
    )

    def __init__(
        self,
        port=None,
        baud_rate=115200,
        telemetry_interval=5.0
    ):
        self.configured_port = port
        self.baud_rate = baud_rate
        self.telemetry_interval = telemetry_interval

        self.connection = None
        self.connected_port = None

        self.receive_buffer = ""

        self.verified = False
        self.ping_sent = False
        self.ping_deadline = 0.0
        self.ready_at = 0.0

        self.next_connect_attempt = 0.0
        self.next_telemetry_request = 0.0

        self.candidate_index = 0

        self.desired_visitor_count = None
        self.last_sent_visitor_count = None

        self.temperature = None
        self.humidity = None

        # Monotonic timestamp used for displaying telemetry age.
        self.telemetry_received_at = None

        # Newly received telemetry samples waiting to be saved.
        self.pending_telemetry_samples = deque()

        self.last_telemetry_error = None

        if not SERIAL_AVAILABLE:
            print(
                "Warning: PySerial is not installed. "
                "ESP32 communication is disabled."
            )

    # --------------------------------------------------------
    # PORT DETECTION
    # --------------------------------------------------------

    def get_candidate_ports(self):
        if not SERIAL_AVAILABLE:
            return []

        if self.configured_port is not None:
            return [self.configured_port]

        detected_ports = list(list_ports.comports())

        if not detected_ports:
            return []

        preferred_keywords = (
            "esp32",
            "cp210",
            "cp210x",
            "ch340",
            "ch341",
            "usb serial",
            "usb uart",
            "silicon labs",
            "wch",
        )

        preferred = []
        remaining = []

        for port in detected_ports:
            description = " ".join([
                str(port.description or ""),
                str(port.manufacturer or ""),
                str(port.product or ""),
                str(port.hwid or "")
            ]).lower()

            if any(
                keyword in description
                for keyword in preferred_keywords
            ):
                preferred.append(port.device)
            else:
                remaining.append(port.device)

        return preferred + remaining

    # --------------------------------------------------------
    # CONNECTION
    # --------------------------------------------------------

    def connect(self):
        if not SERIAL_AVAILABLE:
            return

        candidates = self.get_candidate_ports()

        if not candidates:
            self.next_connect_attempt = (
                time.monotonic()
                + ESP32_RECONNECT_INTERVAL
            )
            return

        port = candidates[
            self.candidate_index % len(candidates)
        ]

        try:
            self.connection = serial.Serial(
                port=port,
                baudrate=self.baud_rate,
                timeout=0,
                write_timeout=0.2
            )

            self.connected_port = port
            self.receive_buffer = ""

            self.verified = False
            self.ping_sent = False
            self.ping_deadline = 0.0

            # Force the current visitor count to be sent after
            # the connection has been verified.
            self.last_sent_visitor_count = None

            current_time = time.monotonic()

            # Opening the serial connection may reset the ESP32.
            self.ready_at = (
                current_time
                + ESP32_BOOT_WAIT
            )

            self.next_telemetry_request = self.ready_at

            try:
                self.connection.reset_input_buffer()

            except (serial.SerialException, OSError):
                pass

            print(
                f"Serial port opened: {self.connected_port}. "
                "Waiting for ESP32..."
            )

        except (
            serial.SerialException,
            OSError,
            PermissionError
        ) as error:
            print(
                f"Could not open serial port {port}: {error}"
            )

            self.connection = None
            self.connected_port = None

            self.candidate_index += 1

            self.next_connect_attempt = (
                time.monotonic()
                + ESP32_RECONNECT_INTERVAL
            )

    def disconnect(self, reason=None):
        if reason:
            print(f"ESP32 disconnected: {reason}")

        if self.connection is not None:
            try:
                self.connection.close()

            except (serial.SerialException, OSError):
                pass

        self.connection = None
        self.connected_port = None

        self.receive_buffer = ""

        self.verified = False
        self.ping_sent = False
        self.last_sent_visitor_count = None

        self.candidate_index += 1

        self.next_connect_attempt = (
            time.monotonic()
            + ESP32_RECONNECT_INTERVAL
        )

    def close(self):
        if self.connection is not None:
            try:
                self.connection.close()

            except (serial.SerialException, OSError):
                pass

        self.connection = None
        self.connected_port = None

    # --------------------------------------------------------
    # SERIAL WRITING
    # --------------------------------------------------------

    def send_command(self, command):
        if self.connection is None:
            return False

        packet = f"<*{command}*>"

        try:
            self.connection.write(
                packet.encode("utf-8")
            )

            return True

        except (
            serial.SerialException,
            serial.SerialTimeoutException,
            OSError
        ) as error:
            self.disconnect(str(error))
            return False

    def send_ping(self):
        if self.send_command("ping"):
            self.ping_sent = True

            self.ping_deadline = (
                time.monotonic()
                + ESP32_HANDSHAKE_TIMEOUT
            )

    def send_visitor_count(self, visitor_count):
        display_count = max(
            0,
            min(
                int(visitor_count),
                MAX_DISPLAY_VISITOR_COUNT
            )
        )

        command = (
            f'people_number:"{display_count}"'
        )

        if self.send_command(command):
            self.last_sent_visitor_count = display_count

    def request_telemetry(self):
        return self.send_command("request_telemetry")

    # --------------------------------------------------------
    # SERIAL READING
    # --------------------------------------------------------

    def read_serial_data(self):
        if self.connection is None:
            return

        try:
            bytes_waiting = self.connection.in_waiting

            if bytes_waiting <= 0:
                return

            received = self.connection.read(
                bytes_waiting
            )

            text = received.decode(
                "utf-8",
                errors="ignore"
            )

            self.receive_buffer += text

        except (
            serial.SerialException,
            OSError
        ) as error:
            self.disconnect(str(error))
            return

        self.process_receive_buffer()

    def process_receive_buffer(self):
        # The current ESP32 ping response is not framed.
        if "Ping_Ack" in self.receive_buffer:
            self.receive_buffer = self.receive_buffer.replace(
                "Ping_Ack",
                ""
            )

            if not self.verified:
                self.verified = True
                self.ping_sent = False

                print(
                    f"ESP32 verified on "
                    f"{self.connected_port}"
                )

        # Process all complete telemetry packets.
        while True:
            match = self.TELEMETRY_PATTERN.search(
                self.receive_buffer
            )

            if match is None:
                break

            try:
                temperature = float(match.group(1))
                humidity = float(match.group(2))

                self.temperature = temperature
                self.humidity = humidity

                # Used for displaying the telemetry age.
                self.telemetry_received_at = (
                    time.monotonic()
                )

                self.last_telemetry_error = None

                # Save a wall-clock timestamp with the sample.
                sample = {
                    "recorded_at_epoch": time.time(),
                    "temperature": temperature,
                    "humidity": humidity
                }

                self.pending_telemetry_samples.append(sample)

                # Prevent unlimited memory usage if database
                # writes continually fail.
                while (
                    len(self.pending_telemetry_samples)
                    > MAX_PENDING_TELEMETRY_SAMPLES
                ):
                    self.pending_telemetry_samples.popleft()

                print(
                    "Telemetry received: "
                    f"{temperature:.1f} C, "
                    f"{humidity:.1f}% humidity"
                )

            except ValueError:
                print(
                    "Received malformed telemetry data."
                )

            self.receive_buffer = (
                self.receive_buffer[:match.start()]
                + self.receive_buffer[match.end():]
            )

        # Process sensor-error responses.
        while True:
            error_match = (
                self.TELEMETRY_ERROR_PATTERN.search(
                    self.receive_buffer
                )
            )

            if error_match is None:
                break

            self.last_telemetry_error = (
                error_match.group(1)
            )

            print(
                "ESP32 telemetry error:",
                self.last_telemetry_error
            )

            self.receive_buffer = (
                self.receive_buffer[:error_match.start()]
                + self.receive_buffer[error_match.end():]
            )

        # Prevent unrelated serial output from growing the
        # receive buffer indefinitely.
        if len(self.receive_buffer) > 1024:
            self.receive_buffer = (
                self.receive_buffer[-512:]
            )

    # --------------------------------------------------------
    # MAIN SERIAL SERVICE
    # --------------------------------------------------------

    def service(self, visitor_count):
        """
        Call once during every main loop iteration.

        The function performs non-blocking serial reads,
        reconnection, telemetry polling and display updates.
        """

        self.desired_visitor_count = max(
            0,
            min(
                int(visitor_count),
                MAX_DISPLAY_VISITOR_COUNT
            )
        )

        if not SERIAL_AVAILABLE:
            return

        current_time = time.monotonic()

        # Try to reconnect periodically.
        if self.connection is None:
            if current_time >= self.next_connect_attempt:
                self.connect()

            return

        # Read serial data without blocking the camera loop.
        self.read_serial_data()

        if self.connection is None:
            return

        # Wait for the ESP32 to boot.
        if current_time < self.ready_at:
            return

        # Verify that the selected serial device is the ESP32.
        if not self.verified:
            if not self.ping_sent:
                self.send_ping()
                return

            if current_time >= self.ping_deadline:
                self.disconnect(
                    "No ping acknowledgement received"
                )

            return

        # Send the visitor count when it changes.
        if (
            self.last_sent_visitor_count
            != self.desired_visitor_count
        ):
            self.send_visitor_count(
                self.desired_visitor_count
            )

            if self.connection is None:
                return

        # Request telemetry periodically.
        if current_time >= self.next_telemetry_request:
            self.request_telemetry()

            self.next_telemetry_request = (
                current_time
                + self.telemetry_interval
            )

    # --------------------------------------------------------
    # STATUS
    # --------------------------------------------------------

    def get_status(self):
        if not SERIAL_AVAILABLE:
            return "ESP32 disabled: PySerial not installed"

        if self.connection is None:
            return "ESP32 not connected"

        if not self.verified:
            return (
                f"ESP32 connecting on "
                f"{self.connected_port}"
            )

        return (
            f"ESP32 connected on "
            f"{self.connected_port}"
        )

    def get_telemetry_age(self):
        if self.telemetry_received_at is None:
            return None

        return (
            time.monotonic()
            - self.telemetry_received_at
        )


# ============================================================
# INITIALIZATION
# ============================================================

model = YOLO(MODEL_PATH)

camera = cv2.VideoCapture(1)

camera.set(
    cv2.CAP_PROP_FRAME_WIDTH,
    CAMERA_WIDTH
)

camera.set(
    cv2.CAP_PROP_FRAME_HEIGHT,
    CAMERA_HEIGHT
)

database = sqlite3.connect(DATABASE_PATH)
cursor = database.cursor()

esp32 = ESP32Bridge(
    port=ESP32_PORT,
    baud_rate=ESP32_BAUD_RATE,
    telemetry_interval=ESP32_TELEMETRY_INTERVAL
)


# ============================================================
# DATABASE SCHEMA
# ============================================================

def ensure_database_column(
    table_name,
    column_name,
    column_declaration
):
    """
    Add a column if it does not already exist.

    The table and column names passed here are constants from
    this program, not external user input.
    """

    cursor.execute(
        f"PRAGMA table_info({table_name})"
    )

    existing_columns = {
        row[1]
        for row in cursor.fetchall()
    }

    if column_name not in existing_columns:
        cursor.execute(
            f"""
            ALTER TABLE {table_name}
            ADD COLUMN {column_name} {column_declaration}
            """
        )

        print(
            f"Added database column: "
            f"{table_name}.{column_name}"
        )


def create_database():
    # Individual completed visits.
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS visits (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            tracking_id INTEGER,
            first_seen TEXT,
            last_seen TEXT,
            duration REAL,
            state TEXT
        )
    """)

    # Periodic temperature, humidity and occupancy samples.
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS telemetry_samples (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            recorded_at TEXT,
            recorded_at_epoch REAL,
            temperature REAL,
            humidity REAL,
            cumulative_visitors INTEGER DEFAULT 0,
            people_visible INTEGER DEFAULT 0,
            visitors_visible INTEGER DEFAULT 0,
            passersby_visible INTEGER DEFAULT 0,
            new_people_visible INTEGER DEFAULT 0
        )
    """)

    # Database migrations for an older telemetry table.
    ensure_database_column(
        "telemetry_samples",
        "recorded_at",
        "TEXT"
    )

    ensure_database_column(
        "telemetry_samples",
        "recorded_at_epoch",
        "REAL"
    )

    ensure_database_column(
        "telemetry_samples",
        "temperature",
        "REAL"
    )

    ensure_database_column(
        "telemetry_samples",
        "humidity",
        "REAL"
    )

    ensure_database_column(
        "telemetry_samples",
        "cumulative_visitors",
        "INTEGER DEFAULT 0"
    )

    ensure_database_column(
        "telemetry_samples",
        "people_visible",
        "INTEGER DEFAULT 0"
    )

    ensure_database_column(
        "telemetry_samples",
        "visitors_visible",
        "INTEGER DEFAULT 0"
    )

    ensure_database_column(
        "telemetry_samples",
        "passersby_visible",
        "INTEGER DEFAULT 0"
    )

    ensure_database_column(
        "telemetry_samples",
        "new_people_visible",
        "INTEGER DEFAULT 0"
    )

    cursor.execute("""
        CREATE INDEX IF NOT EXISTS
        idx_visits_first_seen
        ON visits(first_seen)
    """)

    cursor.execute("""
        CREATE INDEX IF NOT EXISTS
        idx_visits_state
        ON visits(state)
    """)

    cursor.execute("""
        CREATE INDEX IF NOT EXISTS
        idx_telemetry_recorded_at_epoch
        ON telemetry_samples(recorded_at_epoch)
    """)

    database.commit()


# ============================================================
# DATABASE OPERATIONS
# ============================================================

def load_visitor_count():
    cursor.execute("""
        SELECT COUNT(*)
        FROM visits
        WHERE state = 'visitor'
    """)

    result = cursor.fetchone()

    if result is None:
        return 0

    return result[0]


def save_visit(
    person_id,
    person,
    state_name,
    duration
):
    cursor.execute("""
        INSERT INTO visits (
            tracking_id,
            first_seen,
            last_seen,
            duration,
            state
        )
        VALUES (?, ?, ?, ?, ?)
    """, (
        person_id,

        datetime.fromtimestamp(
            person["start_time"]
        ).isoformat(
            sep=" ",
            timespec="seconds"
        ),

        datetime.fromtimestamp(
            person["last_seen"]
        ).isoformat(
            sep=" ",
            timespec="seconds"
        ),

        duration,
        state_name
    ))

    database.commit()


def save_telemetry_sample(
    sample,
    cumulative_visitors,
    people_visible,
    visitors_visible,
    passersby_visible,
    new_people_visible
):
    recorded_at_epoch = sample[
        "recorded_at_epoch"
    ]

    recorded_at = datetime.fromtimestamp(
        recorded_at_epoch
    ).isoformat(
        sep=" ",
        timespec="seconds"
    )

    cursor.execute("""
        INSERT INTO telemetry_samples (
            recorded_at,
            recorded_at_epoch,
            temperature,
            humidity,
            cumulative_visitors,
            people_visible,
            visitors_visible,
            passersby_visible,
            new_people_visible
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        recorded_at,
        recorded_at_epoch,
        float(sample["temperature"]),
        float(sample["humidity"]),
        int(cumulative_visitors),
        int(people_visible),
        int(visitors_visible),
        int(passersby_visible),
        int(new_people_visible)
    ))

    database.commit()


def save_pending_telemetry(
    bridge,
    cumulative_visitors,
    people_visible,
    visitors_visible,
    passersby_visible,
    new_people_visible
):
    """
    Save every telemetry sample received from the ESP32.

    A sample is removed from the queue only after the database
    write succeeds.
    """

    while bridge.pending_telemetry_samples:
        sample = bridge.pending_telemetry_samples[0]

        try:
            save_telemetry_sample(
                sample=sample,
                cumulative_visitors=cumulative_visitors,
                people_visible=people_visible,
                visitors_visible=visitors_visible,
                passersby_visible=passersby_visible,
                new_people_visible=new_people_visible
            )

        except sqlite3.Error as error:
            print(
                "Could not save telemetry sample:",
                error
            )

            # Keep the sample and retry during the next loop.
            break

        else:
            bridge.pending_telemetry_samples.popleft()


# ============================================================
# PERSON STATE
# ============================================================

def create_person(current_time):
    return {
        "start_time": current_time,
        "last_seen": current_time,
        "state": 0
    }


def update_person(person, current_time):
    person["last_seen"] = current_time

    return (
        current_time
        - person["start_time"]
    )


def update_person_state(person, time_seen):
    if (
        PASSERBY_TIME <= time_seen < VISITOR_TIME
        and person["state"] == 0
    ):
        person["state"] = 1

    elif (
        time_seen >= VISITOR_TIME
        and person["state"] != 2
    ):
        person["state"] = 2
        return True

    return False


def get_state_name(state):
    if state == 1:
        return "passerby"

    if state == 2:
        return "visitor"

    return "unknown"


def calculate_visible_counts(
    visible_people_ids,
    people
):
    people_visible = len(visible_people_ids)

    visitors_visible = sum(
        1
        for person_id in visible_people_ids
        if (
            person_id in people
            and people[person_id]["state"] == 2
        )
    )

    passersby_visible = sum(
        1
        for person_id in visible_people_ids
        if (
            person_id in people
            and people[person_id]["state"] == 1
        )
    )

    new_people_visible = sum(
        1
        for person_id in visible_people_ids
        if (
            person_id in people
            and people[person_id]["state"] == 0
        )
    )

    return (
        people_visible,
        visitors_visible,
        passersby_visible,
        new_people_visible
    )


# ============================================================
# DRAWING
# ============================================================

def get_box_color(state):
    if state == 2:
        return (0, 255, 0)      # Green = visitor

    if state == 1:
        return (0, 200, 200)    # Yellow = passerby

    return (0, 0, 250)          # Red = new person


def draw_person(
    frame,
    box,
    person_id,
    time_seen,
    state
):
    x1, y1, x2, y2 = map(
        int,
        box.xyxy[0]
    )

    color = get_box_color(state)

    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        color,
        2
    )

    cv2.putText(
        frame,
        f"ID {person_id}: {time_seen:.1f}s",
        (x1, max(y1 - 10, 20)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2
    )


def draw_visitor_count(
    frame,
    visitor_count,
    people_visible
):
    cv2.putText(
        frame,
        (
            f"{visitor_count} total visitors | "
            f"{people_visible} visible"
        ),
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 0, 250),
        2
    )


def draw_esp32_information(frame, bridge):
    status = bridge.get_status()

    if bridge.verified:
        status_color = (0, 200, 0)
    else:
        status_color = (0, 165, 255)

    cv2.putText(
        frame,
        status,
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        status_color,
        2
    )

    if (
        bridge.temperature is None
        or bridge.humidity is None
    ):
        if bridge.last_telemetry_error:
            telemetry_text = (
                "Telemetry error: "
                f"{bridge.last_telemetry_error}"
            )
        else:
            telemetry_text = "Telemetry: unavailable"

    else:
        age = bridge.get_telemetry_age()

        telemetry_text = (
            f"Telemetry: "
            f"{bridge.temperature:.1f} C, "
            f"{bridge.humidity:.1f}% RH"
        )

        if age is not None:
            telemetry_text += f" ({age:.0f}s ago)"

    cv2.putText(
        frame,
        telemetry_text,
        (10, 85),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )


# ============================================================
# MAIN PROGRAM
# ============================================================

create_database()

visitor_counter = load_visitor_count()
people = {}

print(
    f"Loaded visitor count from database: "
    f"{visitor_counter}"
)


try:
    while True:
        ret, frame = camera.read()

        if not ret:
            print("Could not read from camera.")
            break

        current_time = time.time()

        # ----------------------------------------------------
        # ESP32 COMMUNICATION
        # ----------------------------------------------------

        # This is non-blocking. The application continues even
        # if the ESP32 is unavailable.
        esp32.service(visitor_counter)

        # ----------------------------------------------------
        # YOLO TRACKING
        # ----------------------------------------------------

        results = model.track(
            frame,
            tracker=TRACKER_PATH,
            persist=True,
            classes=[0],
            imgsz=YOLO_IMAGE_SIZE,
            conf=YOLO_CONFIDENCE,
            verbose=False
        )

        annotated_frame = frame.copy()

        # Tracking IDs visible in the current camera frame.
        visible_people_ids = set()

        # ----------------------------------------------------
        # PROCESS DETECTED PEOPLE
        # ----------------------------------------------------

        for box in results[0].boxes:
            if box.id is None:
                continue

            person_id = int(box.id[0])

            visible_people_ids.add(person_id)

            # ------------------------------------------------
            # NEW PERSON
            # ------------------------------------------------

            if person_id not in people:
                people[person_id] = create_person(
                    current_time
                )

            person = people[person_id]

            # ------------------------------------------------
            # UPDATE PERSON
            # ------------------------------------------------

            time_seen = update_person(
                person,
                current_time
            )

            # ------------------------------------------------
            # UPDATE STATE
            # ------------------------------------------------

            became_visitor = update_person_state(
                person,
                time_seen
            )

            if became_visitor:
                visitor_counter += 1

                print(
                    f"VISITOR! "
                    f"ID {person_id} "
                    f"after {time_seen:.1f} seconds"
                )

                # Ask the serial service to send the changed
                # count immediately.
                esp32.service(visitor_counter)

            # ------------------------------------------------
            # DRAW PERSON
            # ------------------------------------------------

            draw_person(
                annotated_frame,
                box,
                person_id,
                time_seen,
                person["state"]
            )

        # ----------------------------------------------------
        # CURRENT OCCUPANCY
        # ----------------------------------------------------

        (
            people_visible,
            visitors_visible,
            passersby_visible,
            new_people_visible
        ) = calculate_visible_counts(
            visible_people_ids,
            people
        )

        # ----------------------------------------------------
        # SAVE NEW TELEMETRY
        # ----------------------------------------------------

        save_pending_telemetry(
            bridge=esp32,
            cumulative_visitors=visitor_counter,
            people_visible=people_visible,
            visitors_visible=visitors_visible,
            passersby_visible=passersby_visible,
            new_people_visible=new_people_visible
        )

        # ----------------------------------------------------
        # FIND PEOPLE WHO HAVE LEFT
        # ----------------------------------------------------

        for person_id in list(people.keys()):
            person = people[person_id]

            time_missing = (
                current_time
                - person["last_seen"]
            )

            if time_missing <= LOST_TIME:
                continue

            # Ignore people who left before reaching the
            # passerby threshold.
            if person["state"] == 0:
                del people[person_id]
                continue

            duration = (
                person["last_seen"]
                - person["start_time"]
            )

            state_name = get_state_name(
                person["state"]
            )

            try:
                save_visit(
                    person_id,
                    person,
                    state_name,
                    duration
                )

            except sqlite3.Error as error:
                print(
                    "Could not save visit:",
                    error
                )

            print()
            print("PERSON LEFT")
            print("ID:", person_id)
            print(
                "Duration:",
                round(duration, 2),
                "seconds"
            )
            print("State:", state_name)
            print()

            del people[person_id]

        # ----------------------------------------------------
        # DRAW INTERFACE
        # ----------------------------------------------------

        draw_visitor_count(
            annotated_frame,
            visitor_counter,
            people_visible
        )

        draw_esp32_information(
            annotated_frame,
            esp32
        )

        # ----------------------------------------------------
        # DISPLAY IMAGE
        # ----------------------------------------------------

        cv2.imshow(
            "YOLO Tracking",
            annotated_frame
        )

        # ----------------------------------------------------
        # QUIT
        # ----------------------------------------------------

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


except KeyboardInterrupt:
    print("\nProgram interrupted by user.")


finally:
    # ========================================================
    # CLEANUP
    # ========================================================

    esp32.close()
    camera.release()
    cv2.destroyAllWindows()
    database.close()

    print("Program closed.")