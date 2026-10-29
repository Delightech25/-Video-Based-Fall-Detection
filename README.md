# -Video-Based-Fall-Detection


A real-time, computer vision-based web application built with **Flask** and **YOLOv8 Pose Estimation** to detect human falls from video uploads.

---

## 📌 Overview

This project provides an automated safety monitoring solution that analyzes video files to recognize human posture and keypoints using Ultralytics YOLOv8. If a fall is detected in the video stream, the system logs the event and alerts the user through an interactive web interface.

---

## ✨ Key Features

* **YOLOv8 Pose Estimation:** Uses `yolov8n-pose.pt` to detect key body joints and analyze spatial orientations/angles.
* **Video Upload & Processing:** Upload `.mp4`, `.avi`, or `.mov` files directly via the Flask web interface for fall analysis.
* **Real-time Alert Dashboard:** View visual overlays of human keypoints and immediate notification when a fall occurs.
* **Database Logging:** Automatically saves session histories and incident logs into SQLite (`database.db`).

---

## 📂 Repository Structure

```text
├── static/              # CSS, JavaScript, and static assets
├── templates/           # HTML templates (Jinja2) for UI layout
├── uploads/             # Directory where uploaded video files are stored
├── app.py               # Main Flask server application logic & routing
├── database.db          # SQLite database for storing logs and user data
├── requirements.txt     # Required Python packages
└── yolov8n-pose.pt      # Pre-trained YOLOv8 pose estimation model weights
🛠️ Tech Stack
Language: Python

Web Framework: Flask

Computer Vision / AI: OpenCV, Ultralytics (YOLOv8 Pose)

Frontend: HTML5, CSS3, JavaScript

Database: SQLite

🚀 Getting Started
1. Prerequisites
Make sure you have Python 3.8+ installed on your system.

2. Installation & Setup
Clone the repository:

Bash
git clone [https://github.com/Delightech25/-Video-Based-Fall-Detection.git](https://github.com/Delightech25/-Video-Based-Fall-Detection.git)
cd -Video-Based-Fall-Detection
Create a virtual environment (recommended):

Bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
Install dependencies:

Bash
pip install -r requirements.txt
Run the Flask application:

Bash
python app.py
Open in Browser:
Navigate to http://127.0.0.1:5000 in your web browser.

🤝 Contributing
Contributions, issues, and feature requests are welcome! Feel free to check out the issues page.

📜 License
This project is open-source and available under the MIT License.


---

### Quick Tip for your repository:
Add a `.gitignore` file to your repository and add `uploads/` and `*.db` to it so that temporary video uploads and local database changes don't clutter your GitHub commits.
