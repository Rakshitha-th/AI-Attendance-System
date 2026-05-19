from flask import Flask, render_template, Response, jsonify
import cv2
import os
import csv
import joblib
import numpy as np
from datetime import datetime
from insightface.app import FaceAnalysis
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

known_embeddings = joblib.load("models/embeddings.pkl")
known_names = joblib.load("models/names.pkl")
normalizer = joblib.load("models/normalizer.pkl")

face_app = FaceAnalysis(name='buffalo_l')
face_app.prepare(ctx_id=-1, det_size=(640, 640))

os.makedirs("attendance", exist_ok=True)
attendance_file = "attendance/attendance.csv"

if not os.path.exists(attendance_file):
    with open(attendance_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Name", "Date", "Time"])

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

face_count = 0

def already_marked_today(name):
    today = datetime.now().strftime("%d-%m-%Y")

    with open(attendance_file, "r") as f:
        reader = csv.reader(f)
        next(reader, None)

        for row in reader:
            if len(row) == 3:
                if row[0] == name and row[1] == today:
                    return True
    return False

def mark_attendance(name):
    if already_marked_today(name):
        return

    date = datetime.now().strftime("%d-%m-%Y")
    time = datetime.now().strftime("%H:%M:%S")

    with open(attendance_file, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([name, date, time])

def generate_frames():
    global face_count

    while True:
        success, frame = cap.read()
        if not success:
            continue

        faces = face_app.get(frame)
        face_count = len(faces)

        for face in faces:
            x1, y1, x2, y2 = map(int, face.bbox)

            emb = normalizer.transform([face.embedding])

            sims = cosine_similarity(emb, known_embeddings)[0]
            idx = np.argmax(sims)
            score = sims[idx]

            name = known_names[idx]

            if score < 0.70:
                name = "Unknown"
                color = (0, 0, 255)
            else:
                color = (0, 255, 0)
                mark_attendance(name)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{name} {score:.2f}",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        color, 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
@app.route('/')
def index():
    attendance = []

    today = datetime.now().strftime("%d-%m-%Y")

    with open(attendance_file, "r") as f:
        reader = csv.reader(f)
        next(reader, None)

        for row in reader:
            if len(row) == 3 and row[1] == today:
                attendance.append(row)

    return render_template(
        "index.html",
        registered_count=len(known_names),
        face_count=face_count,
        attendance=attendance[::-1]
    )


@app.route('/data')
def data():
    attendance = []

    today = datetime.now().strftime("%d-%m-%Y")

    with open(attendance_file, "r") as f:
        reader = csv.reader(f)
        next(reader, None)

        for row in reader:
            if len(row) == 3 and row[1] == today:
                attendance.append(row)

    return jsonify({
        "face_count": face_count,
        "attendance": attendance[::-1]
    })


@app.route('/video')
def video():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=False, threaded=True)