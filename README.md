# 👁️ Biometric AI - Live Camera Face Recognition & Demographics System

A real-time, high-accuracy Computer Vision camera application powered by deep neural networks with ArcFace biometric face recognition, continuous age estimation, gender prediction, 8-class facial emotion analysis, and a futuristic Sci-Fi Biometric HUD overlay.

---

## ✨ Features

- **Live Face Recognition (SFace)**: Recognize registered people with Cosine similarity matching; unknown faces are flagged as `UNREGISTERED`.
- **In-Camera Face Registration**: Press `[R]` while facing the camera to register yourself on-the-fly with a popup name prompt.
- **5-Point Facial Landmarks (YuNet)**: High-speed tracking points on eyes, nose tip, and mouth corners.
- **Continuous Age & Gender Regression**: Realistic single-year age predictions without discrete bracket jumps.
- **8-Class Facial Emotion Analysis (FER+)**: Real-time mood analysis (*Happy, Neutral, Surprise, Sad, Angry, Fear, Disgust, Contempt*).
- **Temporal Anti-Flicker (EMA)**: Exponential Moving Average smoothing eliminates bounding box jitter.
- **Sci-Fi Biometric HUD**: Animated scanner beam, glowing corner brackets, and glassmorphism telemetry cards.
- **Automatic Attendance Logging**: Automatically logs verified detections with timestamps to `logs/attendance_log.csv`.

---

## 🚀 How to Run

Launch the live camera app directly with:

```powershell
python main.py
```
*(or `python run.py`)*

---

## 🎮 Live Camera Keyboard Controls

| Key | Action |
|:---:|---|
| **`[R]`** | **Register Face**: Opens a popup to enter your name and enrolls your face instantly |
| **`[S]`** | **Toggle Scanner**: Turns the animated HUD scanning beam ON / OFF |
| **`[L]`** | **Toggle Landmarks**: Turns 5-point facial landmark tracking dots ON / OFF |
| **`[H]`** | **Toggle HUD**: Shows or hides all telemetry cards |
| **`[C]`** | **Capture**: Saves a snapshot screenshot to `captures/` folder |
| **`[Q]` / `[ESC]`** | **Quit**: Safely closes camera and exits application |

---

## 📁 Project Structure

```
face_recognition_system/
├── main.py                    # Primary Live Camera Application
├── run.py                     # Quick-launch runner
├── core/
│   ├── detector.py            # YuNet face detector & landmark extractor
│   ├── recognizer.py          # SFace deep feature recognizer & database
│   ├── age_gender.py          # Continuous age & gender estimator
│   ├── emotion.py             # FER+ emotion classifier
│   ├── tracker.py             # Multi-face tracker & EMA smoother
│   ├── visualizer.py          # Futuristic Biometric HUD renderer
│   ├── logger.py              # Attendance & security audit logger
│   └── pipeline.py            # Unified biometric pipeline
├── models/                    # Deep learning ONNX model weights
├── known_faces/               # Registered biometric face profiles
├── captures/                  # Saved camera snapshots
├── logs/                      # Attendance and recognition records (CSV)
└── README.md
```