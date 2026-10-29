from flask import Flask, render_template, request, redirect, session, Response, jsonify
import sqlite3
import os
import cv2
import numpy as np
import pandas as pd
from datetime import datetime
from ultralytics import YOLO

app = Flask(__name__)
app.secret_key = "fall_detection_secret"

UPLOAD_FOLDER = "uploads"
STATIC_FOLDER = "static"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ---------------- DATABASE ---------------- #

def init_db():

    conn = sqlite3.connect("database.db")
    c = conn.cursor()

    c.execute("""
    CREATE TABLE IF NOT EXISTS users(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE,
        password TEXT
    )
    """)

    c.execute("""
    CREATE TABLE IF NOT EXISTS detections(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT,
        result TEXT,
        timestamp TEXT,
        confidence INTEGER,
        date TEXT
    )
    """)

    conn.commit()
    conn.close()

init_db()

# ---------------- AI MODEL ---------------- #

model = YOLO("yolov8n-pose.pt")

# ---------------- FALL DETECTION ---------------- #

def detect_fall(video_path, output_path):

    cap = cv2.VideoCapture(video_path)

    width = int(cap.get(3))
    height = int(cap.get(4))
    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps == 0:
        fps = 25

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    prev_center = None
    prev_height = None

    fall_detected = False
    timestamps = []
    confidence = 0

    frame_num = 0

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        frame_num += 1

        results = model(frame, verbose=False)

        for r in results:

            if r.keypoints is None:
                continue

            keypoints = r.keypoints.xy.cpu().numpy()
            boxes = r.boxes.xyxy.cpu().numpy()

            for i, person in enumerate(keypoints):

                shoulder = person[5]
                hip = person[11]

                dx = shoulder[0] - hip[0]
                dy = shoulder[1] - hip[1]

                angle = abs(np.degrees(np.arctan2(dy, dx)))

                x1,y1,x2,y2 = boxes[i]

                height = y2-y1
                width = x2-x1

                ratio = height/width if width>0 else 0

                center = (y1+y2)/2

                speed = 0
                if prev_center is not None:
                    speed = abs(center-prev_center)

                prev_center = center

                height_change = 0
                if prev_height is not None:
                    height_change = abs(height-prev_height)

                prev_height = height

                if angle < 35 and ratio < 1.2 and speed > 20:

                    fall_detected = True

                    sec = int(frame_num/fps)
                    timestamps.append(f"{sec} sec")

                    confidence = min(100,int((speed+height_change+(90-angle))/2))

                    cv2.putText(frame,"FALL DETECTED",
                                (50,70),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1.2,(0,0,255),3)

        out.write(frame)

    cap.release()
    out.release()

    if fall_detected:
        return "Fall Detected", timestamps, confidence
    else:
        return "No Fall Detected", [], 0


# ---------------- LOGIN ---------------- #

@app.route("/", methods=["GET","POST"])
def login():

    if request.method == "POST":

        u = request.form["username"]
        p = request.form["password"]

        conn = sqlite3.connect("database.db")
        c = conn.cursor()

        c.execute("SELECT * FROM users WHERE username=? AND password=?", (u,p))

        user = c.fetchone()

        conn.close()

        if user:
            session["user"] = u
            return redirect("/dashboard")

        return render_template("login.html", error="Invalid login")

    return render_template("login.html")


# ---------------- REGISTER ---------------- #

@app.route("/register", methods=["GET","POST"])
def register():

    if request.method == "POST":

        u = request.form["username"]
        p = request.form["password"]

        try:

            conn = sqlite3.connect("database.db")
            c = conn.cursor()

            c.execute("INSERT INTO users VALUES(NULL,?,?)",(u,p))

            conn.commit()
            conn.close()

            return redirect("/")

        except:

            return render_template("register.html", error="User already exists")

    return render_template("register.html")


# ---------------- DASHBOARD ---------------- #

@app.route("/dashboard")
def dashboard():

    if "user" not in session:
        return redirect("/")

    conn = sqlite3.connect("database.db")
    c = conn.cursor()

    c.execute("SELECT result, COUNT(*) FROM detections GROUP BY result")

    data = c.fetchall()

    conn.close()

    fall = 0
    nofall = 0

    for r in data:

        if r[0] == "Fall Detected":
            fall = r[1]
        else:
            nofall = r[1]

    return render_template("dashboard.html",
                           username=session["user"],
                           fall=fall,
                           nofall=nofall)


# ---------------- REAL TIME DATA ---------------- #

@app.route("/analytics-data")
def analytics_data():

    conn = sqlite3.connect("database.db")
    c = conn.cursor()

    c.execute("SELECT result, COUNT(*) FROM detections GROUP BY result")

    data = c.fetchall()

    conn.close()

    fall = 0
    nofall = 0

    for r in data:
        if r[0] == "Fall Detected":
            fall = r[1]
        else:
            nofall = r[1]

    return jsonify({"fall":fall,"nofall":nofall})


# ---------------- VIDEO UPLOAD ---------------- #

@app.route("/upload", methods=["GET","POST"])
def upload():

    if request.method == "POST":

        file = request.files["video"]

        video_path = os.path.join(UPLOAD_FOLDER,file.filename)
        output_path = "static/output.mp4"

        file.save(video_path)

        result,timestamps,confidence = detect_fall(video_path,output_path)

        conn = sqlite3.connect("database.db")
        c = conn.cursor()

        c.execute("""INSERT INTO detections VALUES(NULL,?,?,?,?,?)""",
                  (session["user"],
                   result,
                   ",".join(timestamps),
                   confidence,
                   datetime.now().strftime("%Y-%m-%d %H:%M:%S")))

        conn.commit()
        conn.close()

        df = pd.DataFrame({
            "User":[session["user"]],
            "Result":[result],
            "Confidence":[confidence],
            "Timestamp":[",".join(timestamps)]
        })

        df.to_csv("static/report.csv", index=False)

        return render_template("upload.html",
                               processed=True,
                               result=result,
                               timestamps=timestamps,
                               confidence=confidence)

    return render_template("upload.html", processed=False)


# ---------------- HISTORY ---------------- #

@app.route("/history")
def history():

    conn = sqlite3.connect("database.db")
    c = conn.cursor()

    c.execute("SELECT * FROM detections")

    data = c.fetchall()

    conn.close()

    return render_template("history.html", data=data)


# ---------------- ANALYTICS ---------------- #

@app.route("/analytics")
def analytics():

    conn = sqlite3.connect("database.db")
    c = conn.cursor()

    c.execute("SELECT result,COUNT(*) FROM detections GROUP BY result")

    data = c.fetchall()

    conn.close()

    fall = 0
    nofall = 0

    for r in data:

        if r[0] == "Fall Detected":
            fall = r[1]
        else:
            nofall = r[1]

    return render_template("analytics.html",
                           fall=fall,
                           nofall=nofall)


# ---------------- WEBCAM STREAM ---------------- #

def generate_frames():

    cap = cv2.VideoCapture(0)

    while True:

        success, frame = cap.read()

        if not success:
            break

        results = model(frame)

        for r in results:

            if r.keypoints is None:
                continue

            keypoints = r.keypoints.xy.cpu().numpy()

            for person in keypoints:

                shoulder = person[5]
                hip = person[11]

                dx = shoulder[0]-hip[0]
                dy = shoulder[1]-hip[1]

                angle = abs(np.degrees(np.arctan2(dy,dx)))

                if angle < 30:

                    cv2.putText(frame,"FALL DETECTED",
                                (50,70),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1.2,(0,0,255),3)

        ret,buffer = cv2.imencode(".jpg",frame)

        frame = buffer.tobytes()

        yield(b'--frame\r\n'
              b'Content-Type: image/jpeg\r\n\r\n'+frame+b'\r\n')


@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route("/webcam")
def webcam():
    return render_template("webcam.html")


# ---------------- ADMIN ---------------- #

@app.route("/admin")
def admin():

    conn = sqlite3.connect("database.db")
    c = conn.cursor()

    c.execute("SELECT * FROM users")
    users = c.fetchall()

    c.execute("SELECT * FROM detections")
    detections = c.fetchall()

    conn.close()

    return render_template("admin.html",
                           users=users,
                           detections=detections)


# ---------------- LOGOUT ---------------- #

@app.route("/logout")
def logout():

    session.clear()
    return redirect("/")


if __name__ == "__main__":
    app.run(debug=True)