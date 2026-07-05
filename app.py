"""
Requires: pip install mediapipe opencv-python numpy ultralytics
Tested against mediapipe==0.10.35 using the modern Tasks API.
"""
import cv2
import numpy as np
from ultralytics import YOLO
import csv
from datetime import datetime
import os
import platform
import threading
import time
import urllib.request

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# =====================================================================
# CONFIGURATION
# =====================================================================
# Alert thresholds
DROWSY_FRAME_LIMIT = 20
SLEEP_FRAME_LIMIT = 45
YAWN_FRAME_LIMIT = 20
DISTRACTION_FRAME_LIMIT = 30

# Eye / Mouth geometry
EAR_THRESHOLD = 0.21
MAR_THRESHOLD = 0.55

# Camera
CAM_WIDTH = 640
CAM_HEIGHT = 480
CAM_FPS = 20

# YOLO
YOLO_INFER_SCALE = 0.4               # further downscale for speed
YOLO_SKIP_FRAMES = 2                 # run YOLO every 3rd frame
YOLO_CONFIDENCE_THRESHOLD = 0.5

# Driver side (for RHD set to 'right', for LHD set to 'left')
DRIVER_SIDE = 'right'                # India = right-hand drive
FLIP_FRAME = True                    # mirror the camera image

# Camera reconnection
MAX_CAMERA_RETRIES = 5

# Phone false‑positive reduction
PHONE_IOU_THRESHOLD = 0.1            # minimum overlap with driver to count as violation

# HUD size and position (top‑right)
HUD_WIDTH = 280
HUD_HEIGHT = 180
HUD_X_OFFSET = 20                    # from right edge
HUD_Y_OFFSET = 20                    # from top

# Logging
LOG_FILE = "event_history_log.csv"

# Non‑blocking audio
BEEP_COOLDOWN_SEC = 1.5
last_beep_time = 0.0

# ---------------------------------------------------------------------
# MediaPipe Tasks model (auto‑download)
MODEL_PATH = "face_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)

# ---------------------------------------------------------------------
# Standard landmark indices (same as mp.solutions.face_mesh)
LEFT_EYE_EAR_IDX = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_EAR_IDX = [33, 160, 158, 133, 153, 144]
MOUTH_TOP_IDX = [82, 13, 312]
MOUTH_BOTTOM_IDX = [87, 14, 317]
MOUTH_LEFT_IDX = 78
MOUTH_RIGHT_IDX = 308
NOSE_TIP_IDX = 1

# =====================================================================
# AUDIO
# =====================================================================
if platform.system() == "Windows":
    import winsound
    def _sync_beep(freq, dur):
        try:
            winsound.Beep(freq, dur)
        except Exception:
            pass
    def alert_beep(frequency, duration):
        global last_beep_time
        now = time.time()
        if now - last_beep_time > max(BEEP_COOLDOWN_SEC, duration / 1000.0):
            last_beep_time = now
            threading.Thread(target=_sync_beep, args=(frequency, duration), daemon=True).start()
else:
    def alert_beep(frequency, duration):
        global last_beep_time
        now = time.time()
        if now - last_beep_time > BEEP_COOLDOWN_SEC:
            last_beep_time = now
            print('\a', end='', flush=True)

# =====================================================================
# HELPER FUNCTIONS
# =====================================================================
def _dist(p1, p2):
    return float(np.hypot(p1[0] - p2[0], p1[1] - p2[1]))

def landmarks_to_px(face_landmarks, w, h):
    return [(lm.x * w, lm.y * h) for lm in face_landmarks]

def eye_aspect_ratio(pts_px, eye_idx):
    p1, p2, p3, p4, p5, p6 = [pts_px[i] for i in eye_idx]
    vertical1 = _dist(p2, p6)
    vertical2 = _dist(p3, p5)
    horizontal = _dist(p1, p4)
    if horizontal == 0:
        return 0.0
    return (vertical1 + vertical2) / (2.0 * horizontal)

def mouth_aspect_ratio(pts_px):
    verticals = [_dist(pts_px[t], pts_px[b]) for t, b in zip(MOUTH_TOP_IDX, MOUTH_BOTTOM_IDX)]
    horizontal = _dist(pts_px[MOUTH_LEFT_IDX], pts_px[MOUTH_RIGHT_IDX])
    if horizontal == 0:
        return 0.0
    return sum(verticals) / (3.0 * horizontal)

def face_bbox_from_landmarks(pts_px, frame_w, frame_h, pad_ratio=0.08):
    xs = [p[0] for p in pts_px]
    ys = [p[1] for p in pts_px]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    pad_x = (x_max - x_min) * pad_ratio
    pad_y = (y_max - y_min) * pad_ratio
    x_min = max(0, int(x_min - pad_x))
    y_min = max(0, int(y_min - pad_y))
    x_max = min(frame_w, int(x_max + pad_x))
    y_max = min(frame_h, int(y_max + pad_y))
    return x_min, y_min, x_max - x_min, y_max - y_min

def box_iou(box1, box2):
    """ box = [x1, y1, x2, y2] """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0

def is_driver_box(box, w_max, driver_side):
    """Return True if the box centre is on the driver's side."""
    cx = (box[0] + box[2]) / 2.0
    if driver_side == 'right':
        return cx > w_max * 0.55
    else:  # 'left'
        return cx < w_max * 0.45

def run_yolo_scaled(frame, scale, conf_threshold):
    if scale < 0.999:
        small = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        inv_scale = 1.0 / scale
    else:
        small = frame
        inv_scale = 1.0
    try:
        result = yolo_model(small, verbose=False)[0]
    except Exception:
        return [], []
    person_boxes = []
    phone_boxes = []
    if result.boxes is not None:
        for item in result.boxes:
            conf = float(item.conf)
            if conf < conf_threshold:
                continue
            label = int(item.cls)
            coords = item.xyxy.flatten().tolist()[:4]
            coords = [c * inv_scale for c in coords]
            if label == 0:
                person_boxes.append(coords)
            elif label == 67:
                phone_boxes.append(coords)
    return person_boxes, phone_boxes

def log_event_to_csv(cabin, identity, event):
    global last_logged_event
    if event != last_logged_event and event != "System Nominal":
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(LOG_FILE, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([timestamp, cabin, identity, event])
        last_logged_event = event

# =====================================================================
# LOAD MODELS
# =====================================================================
print("[INFO] Loading YOLO...")
try:
    yolo_model = YOLO("yolov8n.pt")
except Exception as e:
    print(f"[ERROR] YOLO load failed: {e}")
    exit()

# Download MediaPipe model if missing
if not os.path.exists(MODEL_PATH):
    print("[INFO] Downloading face_landmarker.task...")
    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("[INFO] Model downloaded.")
    except Exception as e:
        print(f"[ERROR] Download failed: {e}\nManually place {MODEL_PATH} next to script.")
        exit()

base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
landmarker_options = mp_vision.FaceLandmarkerOptions(
    base_options=base_options,
    running_mode=mp_vision.RunningMode.VIDEO,
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_face_presence_confidence=0.5,
    min_tracking_confidence=0.5,
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=False,
)
face_landmarker = mp_vision.FaceLandmarker.create_from_options(landmarker_options)

# =====================================================================
# CAMERA SETUP (with auto‑reconnect)
# =====================================================================
def open_camera():
    for attempt in range(MAX_CAMERA_RETRIES):
        if platform.system() == "Windows":
            cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        else:
            cap = cv2.VideoCapture(0)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
            cap.set(cv2.CAP_PROP_FPS, CAM_FPS)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            return cap
        print(f"[WARN] Camera open attempt {attempt+1} failed. Retrying...")
        time.sleep(1)
    return None

cap = open_camera()
if cap is None:
    print("[ERROR] Could not open camera after retries.")
    exit()

# =====================================================================
# FULLSCREEN WINDOW
# =====================================================================
WINDOW_NAME = "Module 8: Unified Dashboard System"
try:
    import tkinter as tk
    _root = tk.Tk()
    screen_w = _root.winfo_screenwidth()
    screen_h = _root.winfo_screenheight()
    _root.destroy()
except Exception:
    screen_w, screen_h = 1920, 1080

cv2.namedWindow(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# =====================================================================
# STATE VARIABLES
# =====================================================================
closed_eye_counter = 0
yawn_counter = 0
distraction_counter = 0
consecutive_face_loss = 0
last_logged_event = ""
last_valid_head_pose = "Center Focus"

cached_person_boxes = []
cached_phone_boxes = []
frame_counter = 0
bad_frame_streak = 0

# Monotonic timestamp for MediaPipe
video_timestamp_ms = 0

# Driver side detection after possible flip
driver_side_in_image = 'left' if (FLIP_FRAME and DRIVER_SIDE == 'right') or (not FLIP_FRAME and DRIVER_SIDE == 'left') else 'right'
# Explanation:
# If FLIP_FRAME=True, left/right are swapped. So if original driver is on right (RHD),
# after flip they appear on left → we detect on left.
# If FLIP_FRAME=False, driver appears on the side they actually sit.

print(f"[INFO] Driver side in image: {driver_side_in_image}")

# =====================================================================
# MAIN LOOP
# =====================================================================
try:
    while cap is not None:
        ret, frame = cap.read()
        if not ret:
            bad_frame_streak += 1
            if bad_frame_streak > 10:
                print("[WARN] Camera lost. Reconnecting...")
                cap.release()
                cap = open_camera()
                if cap is None:
                    print("[ERROR] Reconnection failed. Exiting.")
                    break
                bad_frame_streak = 0
                continue
            continue
        bad_frame_streak = 0

        if FLIP_FRAME:
            frame = cv2.flip(frame, 1)
        h_max, w_max, _ = frame.shape

        # -----------------------------------------------------------------
        # YOLO detection (throttled)
        # -----------------------------------------------------------------
        if frame_counter % (YOLO_SKIP_FRAMES + 1) == 0:
            cached_person_boxes, cached_phone_boxes = run_yolo_scaled(
                frame, YOLO_INFER_SCALE, YOLO_CONFIDENCE_THRESHOLD
            )
        frame_counter += 1

        # -----------------------------------------------------------------
        # Determine driver and occupancy
        # -----------------------------------------------------------------
        driver_box = None
        cabin_occupancy = "Empty Seat"
        driver_classification = "N/A"

        # Find the person most likely to be the driver (by side)
        for box in cached_person_boxes:
            if is_driver_box(box, w_max, driver_side_in_image):
                driver_box = box
                break

        if driver_box is not None:
            x1, y1, x2, y2 = [int(c) for c in driver_box]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            driver_classification = "Driver (Adult)"
            cabin_occupancy = "Driver Active"
        elif cached_person_boxes:
            # There are persons but none on driver side -> passenger
            driver_classification = "Passenger"
            cabin_occupancy = "Passenger Present"

        # -----------------------------------------------------------------
        # Phone detection – only flag if near driver
        # -----------------------------------------------------------------
        phone_present = False
        if driver_box is not None:
            for pbox in cached_phone_boxes:
                if box_iou(driver_box, pbox) > PHONE_IOU_THRESHOLD:
                    phone_present = True
                    cv2.rectangle(frame, (int(pbox[0]), int(pbox[1])),
                                  (int(pbox[2]), int(pbox[3])), (0, 0, 255), 3)
                    break

        # -----------------------------------------------------------------
        # MediaPipe Face Landmarker
        # -----------------------------------------------------------------
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        video_timestamp_ms += 1
        face_detected = False
        face_landmarks = None
        try:
            mesh_result = face_landmarker.detect_for_video(mp_image, video_timestamp_ms)
            if mesh_result.face_landmarks:
                face_detected = True
                face_landmarks = mesh_result.face_landmarks[0]
        except Exception as e:
            # MediaPipe error – treat as no face
            pass

        # -----------------------------------------------------------------
        # Process face landmarks
        # -----------------------------------------------------------------
        eye_status = "Scanning"
        head_pose = "Center Focus"
        yawn_status = "Normal"
        if face_detected and face_landmarks is not None:
            consecutive_face_loss = 0
            pts_px = landmarks_to_px(face_landmarks, w_max, h_max)

            # Bounding box
            fx, fy, fw, fh = face_bbox_from_landmarks(pts_px, w_max, h_max)
            cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh), (255, 255, 0), 2)

            # Head pose (side)
            nose_x = pts_px[NOSE_TIP_IDX][0]
            rel_nose_x = (nose_x - fx) / fw if fw > 0 else 0.5
            if rel_nose_x < 0.40:
                head_pose = "Looking Right"
                last_valid_head_pose = "Looking Right"
                distraction_counter += 1
            elif rel_nose_x > 0.60:
                head_pose = "Looking Left"
                last_valid_head_pose = "Looking Left"
                distraction_counter += 1
            else:
                head_pose = "Center Focus"
                last_valid_head_pose = "Center Focus"
                distraction_counter = max(0, distraction_counter - 2)

            # EAR
            left_ear = eye_aspect_ratio(pts_px, LEFT_EYE_EAR_IDX)
            right_ear = eye_aspect_ratio(pts_px, RIGHT_EYE_EAR_IDX)
            avg_ear = (left_ear + right_ear) / 2.0
            if avg_ear < EAR_THRESHOLD:
                closed_eye_counter += 1
                eye_status = f"Closed (EAR {avg_ear:.2f})"
            else:
                closed_eye_counter = max(0, closed_eye_counter - 3)
                eye_status = f"Open (EAR {avg_ear:.2f})"

            # MAR
            mar = mouth_aspect_ratio(pts_px)
            if mar > MAR_THRESHOLD:
                yawn_counter += 1
                yawn_status = f"Yawning (MAR {mar:.2f})"
                cv2.putText(frame, "YAWN", (fx, max(25, fy - 25)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                yawn_counter = max(0, yawn_counter - 2)
                yawn_status = f"Normal (MAR {mar:.2f})"
        else:
            consecutive_face_loss += 1
            if consecutive_face_loss <= 5:
                head_pose = last_valid_head_pose
                eye_status = "Searching..."
            elif last_valid_head_pose in ["Looking Left", "Looking Right"]:
                head_pose = last_valid_head_pose
                distraction_counter += 1
                closed_eye_counter = max(0, closed_eye_counter - 1)
                eye_status = "Face Turned Away"
            else:
                head_pose = "Face Lost"
                eye_status = "Unknown"
                closed_eye_counter = max(0, closed_eye_counter - 1)

        # -----------------------------------------------------------------
        # Alert arbitration (with priority)
        # -----------------------------------------------------------------
        violations = []
        # Sleep (highest)
        if closed_eye_counter >= SLEEP_FRAME_LIMIT:
            violations.append(("CRITICAL: DRIVER SLEEP DETECTED", (0, 0, 255), 2200, 400))
        # Phone
        elif phone_present:
            violations.append(("MOBILE PHONE USAGE DETECTED", (0, 0, 255), 1400, 150))
        # Drowsy
        elif closed_eye_counter >= DROWSY_FRAME_LIMIT:
            violations.append(("WARNING: DROWSINESS DETECTED", (0, 165, 255), 900, 200))
        # Yawn
        elif yawn_counter >= YAWN_FRAME_LIMIT:
            violations.append(("WARNING: YAWNING DETECTED", (0, 165, 255), 700, 250))
        # Distraction
        elif distraction_counter >= DISTRACTION_FRAME_LIMIT:
            violations.append(("PLEASE FOCUS ON THE ROAD", (0, 165, 255), 550, 200))

        if violations:
            active_alert, alert_color, beep_freq, beep_dur = violations[0]
            alert_beep(beep_freq, beep_dur)
        else:
            active_alert = "System Nominal"
            alert_color = (0, 255, 0)
            last_logged_event = ""   # reset so next event logs again

        log_event_to_csv(cabin_occupancy, driver_classification, active_alert)

        # -----------------------------------------------------------------
        # HUD (top‑right, smaller)
        # -----------------------------------------------------------------
        hud_x = w_max - HUD_WIDTH - HUD_X_OFFSET
        hud_y = HUD_Y_OFFSET
        cv2.rectangle(frame, (hud_x, hud_y),
                      (hud_x + HUD_WIDTH, hud_y + HUD_HEIGHT),
                      (35, 35, 35), cv2.FILLED)
        cv2.rectangle(frame, (hud_x, hud_y),
                      (hud_x + HUD_WIDTH, hud_y + HUD_HEIGHT),
                      (120, 120, 120), 2)

        metrics = [
            f"Cabin : {cabin_occupancy}",
            f"Driver: {driver_classification}",
            f"Eye   : {eye_status}",
            f"Pose  : {head_pose} [{distraction_counter}]",
            f"Phone : {'VIOLATION' if phone_present else 'Clean'}",
            f"Yawn  : {yawn_status}",
            f"Closed: {closed_eye_counter}",
            f"Face  : {consecutive_face_loss}",
        ]

        y_pos = hud_y + 25
        for metric in metrics:
            cv2.putText(frame, metric, (hud_x + 10, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            y_pos += 22

        # -----------------------------------------------------------------
        # Bottom banner
        # -----------------------------------------------------------------
        banner_y1 = max(10, h_max - 60)
        banner_y2 = h_max - 10
        cv2.rectangle(frame, (10, banner_y1), (w_max - 10, banner_y2),
                      alert_color, cv2.FILLED)
        text_size = cv2.getTextSize(active_alert, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        text_x = max(20, (w_max - text_size[0]) // 2)
        text_y = banner_y1 + (banner_y2 - banner_y1 + text_size[1]) // 2
        cv2.putText(frame, active_alert, (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        # -----------------------------------------------------------------
        # Fullscreen letterbox
        # -----------------------------------------------------------------
        scale = min(screen_w / w_max, screen_h / h_max)
        new_w, new_h = int(w_max * scale), int(h_max * scale)
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        display = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
        y_off = (screen_h - new_h) // 2
        x_off = (screen_w - new_w) // 2
        display[y_off:y_off + new_h, x_off:x_off + new_w] = resized

        cv2.imshow(WINDOW_NAME, display)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
    try:
        face_landmarker.close()
    except Exception:
        pass
    print(f"[INFO] Log saved to '{LOG_FILE}'.")