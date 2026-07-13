import cv2
from deepface import DeepFace   #DeepLearning Model for facial recognition
import pandas as pd
from datetime import datetime

cap = cv2.VideoCapture(0)  # try 1 if not working--- 0 is for webcam in laptop
data = []

while True:
    ret, frame = cap.read()

    if not ret:
        print("Camera not working ❌")
        break

    result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
    emotion = result[0]['dominant_emotion']

    print(emotion)  # debug output

    data.append([datetime.now(), emotion])

    cv2.putText(frame, emotion, (50,50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow("Emotion Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

df = pd.DataFrame(data, columns=["Time", "Emotion"])
df.to_csv("emotion_data.csv", index=False)
