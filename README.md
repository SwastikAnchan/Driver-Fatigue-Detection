## Driver Drowsiness and Distraction Detection System

## Overview

This project is an AI-powered Driver Monitoring System (DMS) that detects:

* Driver drowsiness using Eye Aspect Ratio (EAR)
* Driver sleep detection
* Yawning detection using Mouth Aspect Ratio (MAR)
* Head pose distraction detection
* Mobile phone usage while driving
* Driver occupancy monitoring
* Real-time audio alerts
* Event logging to CSV

The system combines:

* MediaPipe Face Landmarker (Tasks API)
* YOLOv8 Object Detection
* OpenCV Computer Vision
* NumPy Mathematical Operations

---

## Features

### Driver Safety Monitoring

✔ Eye Blink Detection

✔ Drowsiness Detection

✔ Sleep Detection

✔ Yawning Detection

✔ Head Pose Tracking

✔ Distraction Monitoring

✔ Mobile Phone Usage Detection

✔ Cabin Occupancy Detection

✔ Event Logging

✔ Audio Alerts

✔ Real-Time Dashboard HUD

---

## System Requirements

### Hardware

* Webcam (USB or Laptop Camera)
* Intel i3 or above
* Minimum 8 GB RAM
* Windows 10/11 Recommended

### Tested Configuration

* Lenovo ThinkPad 11e
* Intel Celeron Processor
* 4GB RAM
* Integrated Webcam

---

## Python Version

Recommended:

```bash
Python 3.10.x
```

Tested and compatible:

```bash
Python 3.10.11
```

Avoid:

```bash
Python 3.12+
```

because some MediaPipe builds may have compatibility issues.

---

## Required Packages

Install the following versions:

```txt
opencv-python==4.10.0.84
numpy==1.26.4
mediapipe==0.10.35
ultralytics==8.3.0
```

---


## Install Dependencies

```bash
pip install opencv-python==4.10.0.84
pip install numpy==1.26.4
pip install mediapipe==0.10.35
pip install ultralytics==8.3.0
```

Or install everything together:

```bash
pip install opencv-python==4.10.0.84 numpy==1.26.4 mediapipe==0.10.35 ultralytics==8.3.0
```

---

## Verify Installation

```bash
pip list
```

Expected:

```txt
mediapipe     0.10.35
opencv-python 4.10.0.84
numpy         1.26.4
ultralytics   8.3.0
```

---

## Project Structure

```txt
Project/
│
├── app.py
├── yolov8n.pt
├── face_landmarker.task
├── event_history_log.csv
└── README.md
```

---

## YOLO Model

The project uses:

```txt
YOLOv8 Nano
```

Model:

```txt
yolov8n.pt
```

The model automatically downloads on first run if not present.

---

## MediaPipe Model

The application automatically downloads:

```txt
face_landmarker.task
```

Model URL:

https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task

---

## Run the Application

```bash
python app.py
```

---

## Controls

Press:

```txt
Q
```

to quit the application.

---

## Detection Thresholds

### Eye Aspect Ratio

```python
EAR_THRESHOLD = 0.21
```

### Mouth Aspect Ratio

```python
MAR_THRESHOLD = 0.55
```

### Drowsiness

```python
DROWSY_FRAME_LIMIT = 20
```

### Sleep

```python
SLEEP_FRAME_LIMIT = 45
```

### Yawn

```python
YAWN_FRAME_LIMIT = 20
```

### Distraction

```python
DISTRACTION_FRAME_LIMIT = 30
```

---

## Generated Logs

The system automatically creates:

```txt
event_history_log.csv
```

Example:

```csv
Timestamp,Cabin,Driver,Event
2026-07-01 10:25:00,Driver Active,Driver (Adult),WARNING: DROWSINESS DETECTED
2026-07-01 10:28:00,Driver Active,Driver (Adult),MOBILE PHONE USAGE DETECTED
```

---

## Detection Pipeline

1. Camera Capture
2. YOLOv8 Person Detection
3. YOLOv8 Phone Detection
4. MediaPipe Face Landmark Detection
5. EAR Calculation
6. MAR Calculation
7. Head Pose Estimation
8. Alert Generation
9. Event Logging
10. Dashboard Display

---

## Performance

Typical Performance:

| Hardware              | FPS       |
| --------------------- | --------- |
| Intel Celeron 4GB RAM | 8–15 FPS  |
| Intel i3 8GB RAM      | 15–25 FPS |
| Intel i5/i7           | 20–35 FPS |

---

## Troubleshooting

### Camera Not Opening

Check:

```bash
Device Manager → Cameras
```

and ensure webcam permissions are enabled.

### MediaPipe Errors

Reinstall:

```bash
pip uninstall mediapipe
pip install mediapipe==0.10.35
```

### YOLO Model Missing

Delete old model and run again:

```bash
yolov8n.pt
```

The model will download automatically.

---

## Future Improvements

* Driver Identification
* Multiple Face Tracking
* Seat Belt Detection
* Smoking Detection
* Cloud Database Logging
* Android Integration
* Raspberry Pi Deployment

---

## Author

AI-Based Driver Drowsiness and Distraction Detection System

Built using:

* Python
* OpenCV
* MediaPipe Tasks API
* YOLOv8
* NumPy
