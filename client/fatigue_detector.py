import cv2
import mediapipe as mp
import numpy as np
from datetime import datetime
import pandas as pd
import time

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()
cap = cv2.VideoCapture(0)
data = []

# EAR calculation function
def eye_aspect_ratio(landmarks):
    left = np.linalg.norm(landmarks[159] - landmarks[145])
    right = np.linalg.norm(landmarks[386] - landmarks[374])
    return (left + right) / 2

# MAR calculation function
def mouth_ratio(landmarks):
    top = np.linalg.norm(landmarks[13] - landmarks[14])
    width = np.linalg.norm(landmarks[78] - landmarks[308])
    return top / width

# EAR calibration function
def calibrate_ear(cap, face_mesh, seconds=5):
    print("[INFO] Calibrating EAR... Please stay alert and look at the screen.")
    time.sleep(2)  # small pause before starting
    ear_values = []
    start_time = time.time()

    while time.time() - start_time < seconds:
        ret, frame = cap.read()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = face_mesh.process(rgb)

        if result.multi_face_landmarks:
            for face_landmarks in result.multi_face_landmarks:
                h, w, _ = frame.shape
                landmarks = np.array([(lm.x * w, lm.y * h) for lm in face_landmarks.landmark])
                ear = eye_aspect_ratio(landmarks)
                ear_values.append(ear)

        # Optional: show calibration progress
        cv2.putText(frame, "Calibrating... Please stay alert", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.imshow('Calibration', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyWindow('Calibration')
    if ear_values:
        avg_ear = np.mean(ear_values)
        print(f"[INFO] Calibration complete. Baseline EAR = {avg_ear:.2f}")
        return avg_ear
    else:
        print("[WARNING] Calibration failed. Defaulting EAR threshold.")
        return 15  # fallback

# ===== MAIN LOGIC =====

baseline_ear = calibrate_ear(cap, face_mesh)
ear_threshold = baseline_ear * 0.7  # 70% of normal EAR

while True:
    ret, frame = cap.read()
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb)

    if result.multi_face_landmarks:
        for face_landmarks in result.multi_face_landmarks:
            h, w, _ = frame.shape
            landmarks = np.array([(lm.x * w, lm.y * h) for lm in face_landmarks.landmark])

            ear = eye_aspect_ratio(landmarks)
            mar = mouth_ratio(landmarks)

            state = "Alert"
            if ear < ear_threshold:
                state = "Drowsy"
            elif mar > 0.4:
                state = "Yawning"

            data.append([datetime.now(), ear, mar, state])
            cv2.putText(frame, f"Status: {state}", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow('Fatigue Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Save session data
pd.DataFrame(data, columns=["Timestamp", "EAR", "MAR", "State"]).to_csv("wfh_fatigue_data.csv", index=False)
print("[INFO] Session data saved to wfh_fatigue_data.csv")
