import customtkinter as ctk
from PIL import Image, ImageTk
import cv2
import numpy as np
import joblib
import os
import tkinter as tk

from datetime import datetime
from insightface.app import FaceAnalysis
from sklearn.metrics.pairwise import cosine_similarity
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("dark-blue")
app = ctk.CTk()

app.title("AI Face Recognition Attendance System")
app.state("zoomed")

app.configure(fg_color="#0f172a")


known_embeddings = joblib.load("models/embeddings.pkl")
known_names = joblib.load("models/names.pkl")
normalizer = joblib.load("models/normalizer.pkl")


face_app = FaceAnalysis(name='buffalo_l')


face_app.prepare(ctx_id=-1)

os.makedirs("attendance", exist_ok=True)

date_today = datetime.now().strftime("%Y-%m-%d")

attendance_file = f"attendance/attendance_{date_today}.csv"

# create attendance file
if not os.path.exists(attendance_file):

    with open(attendance_file, "w") as f:
        f.write("Name,Time\n")

marked_names = set()

# LEFT SIDEBAR
left_frame = ctk.CTkFrame(app,width=260,corner_radius=20,fg_color="#111827"
)

left_frame.pack(side="left",fill="y",padx=15,pady=15
)

# CENTER FRAME
center_frame = ctk.CTkFrame(app,corner_radius=20,fg_color="#111827"
)

center_frame.pack(side="left",fill="both",expand=True,padx=10,pady=15
)

# RIGHT FRAME
right_frame = ctk.CTkFrame(app,width=320,corner_radius=20,fg_color="#111827"
)

right_frame.pack(side="right",fill="y",padx=15,pady=15
)

logo = ctk.CTkLabel(left_frame,text="FaceLog AI",font=("Arial", 34, "bold"),text_color="#60a5fa"
)

logo.pack(pady=(25, 5))

subtitle = ctk.CTkLabel(left_frame,text="FACE RECOGNITION\nATTENDANCE SYSTEM",font=("Arial", 14),text_color="gray"
)

subtitle.pack(pady=(0, 25))

clock_label = ctk.CTkLabel(left_frame,text="",font=("Arial", 20, "bold"),text_color="white"
)

clock_label.pack(pady=15)

def update_clock():

    now = datetime.now().strftime("%I:%M:%S %p\n%d-%m-%Y")
    clock_label.configure(text=now)

    clock_label.after(1000, update_clock)

update_clock()


registered_card = ctk.CTkFrame(left_frame,fg_color="#1e293b",corner_radius=18
)

registered_card.pack(padx=15,pady=10,fill="x"
)

registered_count = len(known_names)

registered_label = ctk.CTkLabel(
    registered_card,
    text=f"{registered_count}\nREGISTERED",
    font=("Arial", 26, "bold"),
    text_color="white"
)

registered_label.pack(pady=25)

present_card = ctk.CTkFrame(
    left_frame,fg_color="#1e293b",corner_radius=18
)

present_card.pack(padx=15,pady=10,fill="x"
)

present_label = ctk.CTkLabel(present_card,text="0\nPRESENT TODAY",font=("Arial", 26, "bold"),text_color="white"
)

present_label.pack(pady=25)

# FACE COUNT CARD

face_card = ctk.CTkFrame(left_frame,fg_color="#1e293b",corner_radius=18
)

face_card.pack(padx=15,pady=10,fill="x"
)

face_count_label = ctk.CTkLabel(face_card,text="0\nFACES DETECTED",font=("Arial", 24, "bold"),text_color="white"
)

face_count_label.pack(pady=25)

# CAMERA STATUS

status_label = ctk.CTkLabel(left_frame,text="● CAMERA ACTIVE",font=("Arial", 16, "bold"),text_color="#22c55e"
)

status_label.pack(pady=25)

def open_attendance():

    os.startfile("attendance")

attendance_btn = ctk.CTkButton(left_frame,text="Open Attendance",height=45,corner_radius=12,font=("Arial", 16, "bold"),fg_color="#2563eb",hover_color="#1d4ed8",
    command=open_attendance
)

attendance_btn.pack(padx=20,pady=(20, 10),fill="x"
)

# EXIT BUTTON

def close_app():
    cap.release()
    app.destroy()
exit_btn = ctk.CTkButton(left_frame,text="Exit",height=45,corner_radius=12,font=("Arial", 16, "bold"),fg_color="#dc2626",hover_color="#b91c1c",command=close_app
)

exit_btn.pack(padx=20,pady=10,fill="x"
)

# CENTER PANEL

camera_title = ctk.CTkLabel(
    center_frame,
    text="LIVE FACE RECOGNITION",
    font=("Arial", 28, "bold"),
    text_color="white"
)

camera_title.pack(pady=15)
camera_label = ctk.CTkLabel(
    center_frame,
    text=""
)
camera_label.pack(pady=10)

# RIGHT PANEL

logs_title = ctk.CTkLabel(
    right_frame,
    text="Today's Attendance",
    font=("Arial", 28, "bold"),
    text_color="white"
)

logs_title.pack(pady=20)

log_box = ctk.CTkTextbox(right_frame,width=280,height=750,corner_radius=15,fg_color="#0f172a",font=("Consolas", 15),text_color="white"
)

log_box.pack(
    padx=15,
    pady=10
)
cap = cv2.VideoCapture(0)
def refresh_present_count():
    count = len(marked_names)
    present_label.configure(
        text=f"{count}\nPRESENT TODAY"
    )

def update_camera():
    ret, frame = cap.read()

    if ret:

        faces = face_app.get(frame)
        # face count
        face_count_label.configure(
            text=f"{len(faces)}\nFACES DETECTED"
        )

        for face in faces:

            x1, y1, x2, y2 = map(int, face.bbox)
            embedding = face.embedding.reshape(1, -1)
            embedding = normalizer.transform(embedding)
            similarities = cosine_similarity(
                embedding,
                known_embeddings
            )[0]

            best_index = np.argmax(similarities)
            best_similarity = similarities[best_index]

            name = known_names[best_index]
            if best_similarity < 0.65:
                name = "Unknown"

                color = (0, 0, 255)

                cv2.putText(
                    frame,
                    "Unknown",
                    (x1, y2 + 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2
                )

            else:

                color = (0, 255, 120)
                if name not in marked_names:
                    marked_names.add(name)
                    refresh_present_count()
                    current_time = datetime.now().strftime("%H:%M:%S")

                    # LOG BOX

                    log_box.insert(
                        "end",
                        f"{name:<15} {current_time}\n"
                    )
                    log_box.see("end")

                    # SAVE CSV
                    with open(attendance_file, "a") as f:
                        f.write(
                            f"{name},{current_time}\n"
                        )
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                color,
                3
            )

            text = f"{name} ({best_similarity:.2f})"
            cv2.putText(
                frame,
                text,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2
            )
        frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        img = Image.fromarray(frame)
        img = img.resize((950, 700))
        imgtk = ImageTk.PhotoImage(image=img)
        camera_label.imgtk = imgtk
        camera_label.configure(image=imgtk)

    camera_label.after(10, update_camera)
update_camera()
app.mainloop()