# 🔫 Weapon Detection using YOLOv8

An AI-powered, real-time weapon detection system that identifies **firearms and knives** in images, videos, and live camera feeds using **YOLOv8**, with a **Flask** backend and a lightweight web interface.

Built as a final-year MSc (Data Analytics) project at Pillai College of Arts, Commerce and Science (Autonomous), New Panvel — University of Mumbai, 2024–2025.

---

## 📖 Overview

Traditional security systems rely heavily on manual CCTV monitoring, which is prone to human fatigue, delayed response, and oversight. This project automates that process using deep learning–based object detection to flag firearms and knives the moment they appear in an image, video, or live webcam feed — surfacing real-time visual alerts instead of relying on a person watching a screen.

The system supports three input modes:
- **Image upload** — detect weapons in a single photo
- **Video upload** — detect weapons frame-by-frame in recorded footage
- **Live camera feed** — real-time detection via webcam with on-screen alerts

---

## ✨ Features

- Real-time weapon detection (guns & knives) using a fine-tuned **YOLOv8** model
- Supports image, video, and live webcam input
- Bounding boxes with confidence scores overlaid on detected weapons
- Live on-screen alert banner when a weapon is detected
- Simple, responsive web UI (HTML/CSS/JavaScript)
- File-based result storage — no database required

---

## 🧱 Tech Stack

**Frontend**
- HTML, CSS, JavaScript
- Bootstrap (layout/responsiveness)

**Backend**
- Python, Flask (routing, API endpoints, request handling)
- YOLOv8 (Ultralytics) — object detection model
- OpenCV — video frame processing and detection overlay
- NumPy / Pandas — data handling
- TensorFlow / Keras — supporting model tooling

**Storage**
- No relational database — detection results are stored as image/video files in `uploads/` and `results/` folders

---

## 🏗️ System Architecture

The system follows a simple 3-layer architecture:

```
Presentation Layer   →  HTML / CSS / JS (upload UI, live feed display, alerts)
Application Layer    →  Flask backend, YOLOv8 model inference, API endpoints
Storage Layer         →  File-based storage (uploads/ and results/ folders)
```

**Data flow:** user uploads a file (or starts a live feed) → Flask receives and validates it → YOLOv8 runs inference and draws bounding boxes → result is saved to `results/` → UI displays the processed output with detection confidence scores.

---

## 📂 Dataset & Trained Weights

Due to GitHub's file size limits, the full training dataset and trained model weights are hosted externally instead of in this repository.

📁 **Dataset & Weights (Google Drive):** [Add your Google Drive link here](https://drive.google.com/your-link-here)

> Replace the link above with your actual shareable Google Drive link (Anyone with the link → Viewer).

The dataset was sourced from the *"Weapons and Similar Handled Objects"* dataset (Kaggle), annotated in YOLO format (bounding boxes across `train` / `valid` / `test` splits, 80/10/10), with images standardized to 640×640 per YOLOv8's input requirement.

---

## 📁 Project Structure

```
Weapons-and-Knives-Detector-with-YOLOv8/
├── app.py                  # Flask app entry point
├── data.yaml                # YOLO dataset config (classes, paths)
├── train.py                  # Model training script
├── preprocessing-images.py   # Dataset preprocessing / augmentation
├── detecting-images.py       # Standalone detection script
├── graphs.ipynb               # Training performance visualizations
├── requirements.txt
├── templates/
│   ├── index.html            # Upload page
│   ├── result.html           # Detection result page
│   └── live.html             # Live webcam detection page
├── uploads/                  # User-uploaded files (gitignored)
├── results/                  # Processed detection outputs (gitignored)
└── README.md
```

---

## ⚙️ Installation & Setup

```bash
# 1. Clone the repository
git clone https://github.com/your-username/Weapons-and-Knives-Detector-with-YOLOv8.git
cd Weapons-and-Knives-Detector-with-YOLOv8

# 2. Create a virtual environment (recommended)
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # macOS/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download the trained weights from the Google Drive link above
#    and place best.pt inside runs/detect/Normal_Compressed/weights/

# 5. Run the Flask app
python app.py
```

The app will be available at `http://127.0.0.1:5000`.

---

## 🚀 Usage

- **Upload detection:** Go to the home page → choose an image or video → click **Upload** → view the result with bounding boxes and confidence scores.
- **Live detection:** Click **Live Detection** → grant webcam access → real-time bounding boxes and alerts appear as weapons are detected in the feed.

---

## 🧪 Model Training

The model was trained using `train.py` with the following configuration:

| Parameter | Value |
|---|---|
| Base model | YOLOv8n (Nano) |
| Epochs | 150 |
| Batch size | 6 |
| Image size | 640×640 |
| Optimizer | Adam |
| IoU threshold | 0.7 |

Evaluation was done using **mAP50**, **mAP50-95**, **Precision**, **Recall**, and **F1 Score** via `Performance.ipynb`. See `graphs.ipynb` and the `runs/detect/` output folder for training curves and validation results.

---

## 🔭 Future Scope

- Multi-class weapon classification (handguns, rifles, melee weapons)
- Integration with CCTV/surveillance networks and edge devices (Jetson Nano, Raspberry Pi)
- Automated alert system (email/SMS) linked to security personnel
- Low-light and occlusion handling via infrared/thermal imaging
- Mobile app version using TensorFlow Lite
- Cloud deployment (AWS / GCP / Heroku) with Docker

---

## 🙏 Acknowledgements

Developed by **Jayesh Jadhav** as part of the MSc (Data Analytics) curriculum, Department of Computer Science, Pillai College of Arts, Commerce and Science (Autonomous), New Panvel — under the guidance of **Mr. Omkar Sherkhane**, Assistant Professor.

---

## ⚠️ Disclaimer

This project was built for academic and research purposes to demonstrate real-time object detection techniques. It is a prototype and has not been validated for deployment in live security-critical environments.
