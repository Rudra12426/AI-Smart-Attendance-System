# use your own image dataset fro this project 

# ================== AI SMART ATTENDANCE SYSTEM (MULTI FACE) ==================

import cv2
import os
import csv
from datetime import datetime
from deepface import DeepFace
import re
import winsound

# -------------------------------
# CONFIG
# -------------------------------
DATASET_PATH = r"C:\Users\Victus\OneDrive\Desktop\ML Project\AI Attendance System\Dataset"
UNKNOWN_FOLDER = r"C:\Users\Victus\OneDrive\Desktop\ML Project\AI Attendance System\UnknownFaces"
ATTENDANCE_FILE = r"C:\Users\Victus\OneDrive\Desktop\ML Project\AI Attendance System\attendance.csv"

os.makedirs(UNKNOWN_FOLDER, exist_ok=True)

# -------------------------------
# CREATE CSV IF NOT EXISTS
# -------------------------------
if not os.path.exists(ATTENDANCE_FILE):
    with open(ATTENDANCE_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Name", "Date", "Time"])
    print("📄 attendance.csv created")

marked_names = set()
saved_unknown_faces = 0
MAX_UNKNOWN_SAVE = 5  # prevent spam

# -------------------------------
# LOAD DATASET
# -------------------------------
dataset_images = []

for img in os.listdir(DATASET_PATH):
    path = os.path.join(DATASET_PATH, img)
    if os.path.isfile(path):
        name_match = re.match(r"([A-Za-z]+)", img)
        person_name = name_match.group(1) if name_match else "Unknown"
        dataset_images.append((path, person_name))

print(f"✅ Dataset Loaded: {len(dataset_images)} images")

# -------------------------------
# START CAMERA
# -------------------------------
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
print("🎥 Camera Started | Press ESC to Exit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    try:
        faces = DeepFace.extract_faces(
            img_path=frame,
            detector_backend="retinaface",
            enforce_detection=False
        )
    except:
        faces = []

    for face in faces:
        # ✅ FIXED FACIAL AREA EXTRACTION
        fa = face["facial_area"]
        x = fa["x"]
        y = fa["y"]
        w = fa["w"]
        h = fa["h"]

        face_crop = frame[y:y+h, x:x+w]
        recognized_name = "Unknown"

        # -------------------------------
        # COMPARE WITH DATASET
        # -------------------------------
        for img_path, person_name in dataset_images:
            try:
                result = DeepFace.verify(
                    img1_path=face_crop,
                    img2_path=img_path,
                    model_name="Facenet",
                    detector_backend="retinaface",
                    enforce_detection=False,
                    silent=True
                )

                if result["verified"]:
                    recognized_name = person_name
                    break

            except:
                continue

        # -------------------------------
        # KNOWN FACE
        # -------------------------------
        if recognized_name != "Unknown":
            color = (0, 255, 0)

            if recognized_name not in marked_names:
                winsound.Beep(1200, 200)

                now = datetime.now()
                with open(ATTENDANCE_FILE, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        recognized_name,
                        now.strftime("%Y-%m-%d"),
                        now.strftime("%H:%M:%S")
                    ])

                marked_names.add(recognized_name)
                print(f"✅ Attendance marked: {recognized_name}")

        # -------------------------------
        # UNKNOWN FACE
        # -------------------------------
        else:
            color = (0, 0, 255)

            if saved_unknown_faces < MAX_UNKNOWN_SAVE:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_path = os.path.join(
                    UNKNOWN_FOLDER,
                    f"unknown_{timestamp}.jpg"
                )
                cv2.imwrite(save_path, face_crop)
                winsound.Beep(600, 400)
                saved_unknown_faces += 1
                print(f"⚠ Unknown face saved: {save_path}")

        # -------------------------------
        # DRAW BOX & LABEL
        # -------------------------------
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(
            frame,
            recognized_name,
            (x, y-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            color,
            2
        )

    cv2.imshow("AI Smart Attendance System (Multi-Face)", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
print("🛑 System Closed")



