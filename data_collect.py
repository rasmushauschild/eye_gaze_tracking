import cv2 as cv
import pyautogui
import os
import mediapipe as mp

def CaptureEye(captures=1, frame_scale=0.15, coordinates=(0, 0), folder_name="eyes"):
    os.makedirs(folder_name, exist_ok=True)  # create folder, overwrite existing
    cam = cv.VideoCapture(0)

    face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)  # Initialize face mesh outside the loop

    for i in range(captures):
        ret, frame = cam.read()  # store single webcam frame
        if not ret:
            print("Failed to capture image.")
            continue

        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        result = face_mesh.process(rgb_frame)

        if result.multi_face_landmarks:
            landmarks = result.multi_face_landmarks[0].landmark

            # Ensure the landmarks indices are valid
            eye_box = frame[
                int(landmarks[27].y * frame.shape[0]):int(landmarks[23].y * frame.shape[0]),
                int(landmarks[226].x * frame.shape[1]):int(landmarks[190].x * frame.shape[1])
            ]

            eye_box = cv.cvtColor(eye_box, cv.COLOR_BGR2GRAY)
            eye_box = cv.resize(eye_box, dsize=(100, 50))

            cv.imshow('eye_box', eye_box)
            cv.imwrite(
                os.path.join(folder_name, f"{coordinates[0]}.{coordinates[1]}.{i}.jpg"), eye_box)

            if cv.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            print("No face detected.")

    cam.release()
    cv.destroyAllWindows()
            


for i in [0, 256, 512, 768, 1024, 1280, 1536, 1792, 2048, 2304, 2560]:
    for j in [0, 256, 512, 768, 1024, 1280, 1400]:
        print(i,j)
        pyautogui.moveTo(i, j)
        input("Press Enter to continue...")
        pyautogui.moveTo(i, j)
        CaptureEye(captures=100, frame_scale=0.15, coordinates=(i,j), folder_name="new_eyes")
