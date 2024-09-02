import cv2
import mediapipe as mp

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5, refine_landmarks = True)

# Initialize drawing utilities for landmarks
mp_drawing = mp.solutions.drawing_utils

# Open the webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    # Convert the BGR image to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Process the image and detect landmarks
    results = face_mesh.process(image_rgb)

    # Convert the image color back so it can be displayed
    image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    # Draw landmarks if detected
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Draw the face landmarks on the image
            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=1, circle_radius=1)
            )

            # Print the index of each landmark on the face
            for idx, landmark in enumerate(face_landmarks.landmark):
                h, w, _ = image.shape
                x, y = int(landmark.x * w), int(landmark.y * h)

                # Put the index number of the landmark on the image
                cv2.putText(image, str(idx), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1, cv2.LINE_AA)

    # Display the image
    cv2.imshow('MediaPipe Face Mesh', image)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()
