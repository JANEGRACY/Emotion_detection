import cv2
from deepface import DeepFace
from datetime import datetime
import mysql.connector
import time

# Connect to MySQL
conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="jane123",
    database="emotion_db"
)

cursor = conn.cursor()

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Camera not working!")
    exit()

print("Press 'q' to quit")

last_saved = time.time()

while True:

    ret, frame = cap.read()

    if not ret:
        break

    try:
        result = DeepFace.analyze(
            frame,
            actions=['emotion'],
            enforce_detection=False
        )

        emotion = result[0]['dominant_emotion']

        # Save only once every second
        if time.time() - last_saved >= 1:

            current_time = datetime.now()

            sql = "INSERT INTO emotions(time, emotion) VALUES(%s,%s)"
            cursor.execute(sql, (current_time, emotion))
            conn.commit()

            print(current_time, emotion)

            last_saved = time.time()

        cv2.putText(
            frame,
            emotion,
            (50,50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0,255,0),
            2
        )

    except Exception as e:
        print(e)

    cv2.imshow("Emotion Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

cursor.close()
conn.close()

print("Data saved successfully.")
