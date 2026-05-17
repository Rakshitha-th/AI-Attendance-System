# =========================================================
# ADD THESE IMPORTS AT TOP
# =========================================================

from tkinter import filedialog
import shutil

# =========================================================
# REPLACE THIS
# =========================================================

face_app = FaceAnalysis(name='buffalo_l')

# =========================================================
# WITH THIS
# =========================================================

face_app = FaceAnalysis(name='buffalo_s')

# =========================================================
# ADD THIS BELOW IMPORTS
# =========================================================

cv2.setUseOptimized(True)

# =========================================================
# REPLACE REGISTER STUDENT FUNCTION
# =========================================================

def register_student():

    global known_embeddings
    global known_names

    student_name = name_entry.get().strip()

    if student_name == "":
        return

    # ============================================
    # SELECT IMAGES
    # ============================================

    file_paths = filedialog.askopenfilenames(
        title="Select Student Images",
        filetypes=[
            ("Image Files", "*.jpg *.jpeg *.png")
        ]
    )

    if len(file_paths) == 0:
        return

    # ============================================
    # CREATE STUDENT FOLDER
    # ============================================

    student_path = os.path.join(
        "Dataset",
        student_name
    )

    os.makedirs(student_path, exist_ok=True)

    embeddings = []

    # ============================================
    # COPY IMAGES
    # ============================================

    for idx, file_path in enumerate(file_paths):

        extension = os.path.splitext(file_path)[1]

        save_path = os.path.join(
            student_path,
            f"{idx}{extension}"
        )

        shutil.copy(
            file_path,
            save_path
        )

        # ========================================
        # CREATE EMBEDDINGS
        # ========================================

        img = cv2.imread(save_path)

        if img is None:
            continue

        try:

            faces = face_app.get(img)

            if len(faces) == 0:
                continue

            embedding = faces[0].embedding

            embeddings.append(embedding)

        except:
            continue

    if len(embeddings) == 0:

        print("No face detected")

        return

    # ============================================
    # MEAN EMBEDDING
    # ============================================

    mean_embedding = np.mean(
        embeddings,
        axis=0
    )

    mean_embedding = mean_embedding.reshape(1, -1)

    mean_embedding = normalizer.transform(
        mean_embedding
    )

    # ============================================
    # UPDATE DATA
    # ============================================

    known_embeddings = np.vstack(
        [known_embeddings, mean_embedding]
    )

    known_names.append(student_name)

    # ============================================
    # SAVE UPDATED MODELS
    # ============================================

    joblib.dump(
        known_embeddings,
        "models/embeddings.pkl"
    )

    joblib.dump(
        known_names,
        "models/names.pkl"
    )

    # ============================================
    # REFRESH UI
    # ============================================

    registered_label.configure(
        text=f"{len(known_names)}\nREGISTERED"
    )

    refresh_dataset()

    name_entry.delete(0, "end")

    print(f"{student_name} Added Successfully")

# =========================================================
# DELETE STUDENT FUNCTION
# =========================================================

def delete_student():

    global known_embeddings
    global known_names

    student_name = name_entry.get().strip()

    if student_name == "":
        return

    # ============================================
    # REMOVE DATASET FOLDER
    # ============================================

    student_path = os.path.join(
        "Dataset",
        student_name
    )

    if os.path.exists(student_path):

        shutil.rmtree(student_path)

    # ============================================
    # REMOVE EMBEDDING
    # ============================================

    if student_name in known_names:

        index = known_names.index(student_name)

        known_names.pop(index)

        known_embeddings = np.delete(
            known_embeddings,
            index,
            axis=0
        )

        # SAVE UPDATED FILES

        joblib.dump(
            known_embeddings,
            "models/embeddings.pkl"
        )

        joblib.dump(
            known_names,
            "models/names.pkl"
        )

    # ============================================
    # REFRESH UI
    # ============================================

    registered_label.configure(
        text=f"{len(known_names)}\nREGISTERED"
    )

    refresh_dataset()

    name_entry.delete(0, "end")

    print(f"{student_name} Deleted Successfully")

# =========================================================
# ADD THIS BELOW SAVE STUDENT BUTTON
# =========================================================

delete_btn = ctk.CTkButton(
    left_frame,
    text="Delete Student",
    height=42,
    corner_radius=12,
    fg_color="#dc2626",
    hover_color="#b91c1c",
    font=("Arial", 15, "bold"),
    command=delete_student
)

delete_btn.pack(
    padx=20,
    pady=5,
    fill="x"
)

# =========================================================
# REPLACE CAMERA SIZE
# =========================================================

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

# =========================================================
# ADD THIS ABOVE update_camera()
# =========================================================

frame_count = 0

# =========================================================
# REPLACE COMPLETE update_camera() FUNCTION
# =========================================================

def update_camera():

    global frame_count

    ret, frame = cap.read()

    if not ret:
        return

    frame_count += 1

    # PROCESS EVERY 3RD FRAME
    if frame_count % 3 == 0:

        faces = face_app.get(frame)

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

            # ====================================
            # UNKNOWN
            # ====================================

            if best_similarity < 0.65:

                name = "Unknown"

                color = (80, 80, 255)

            else:

                color = (0, 215, 255)

                if name not in marked_names:

                    marked_names.add(name)

                    refresh_present_count()

                    current_time = datetime.now().strftime("%H:%M:%S")

                    log_box.insert(
                        "end",
                        f"{name:<15} {current_time}\n"
                    )

                    log_box.see("end")

                    with open(attendance_file, "a") as f:

                        f.write(
                            f"{name},{current_time}\n"
                        )

            # ====================================
            # FACE BOX
            # ====================================

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                color,
                2
            )

            cv2.putText(
                frame,
                name,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2
            )

    # ============================================
    # SHOW FRAME
    # ============================================

    frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    img = Image.fromarray(frame)

    img = img.resize((850, 600))

    imgtk = ImageTk.PhotoImage(image=img)

    camera_label.imgtk = imgtk

    camera_label.configure(image=imgtk)

    camera_label.after(1, update_camera)