import torch
import pyautogui
from torchvision import transforms
import cv2 as cv
import mediapipe as mp
from PIL import Image

# Define the EyeGazeCNN model (ensure this matches the structure of your trained model)
class EyeGazeCNN(torch.nn.Module):
    def __init__(self):
        super(EyeGazeCNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = torch.nn.Linear(64 * 25 * 12, 512)  # Adjust based on input size after pooling
        self.fc2 = torch.nn.Linear(512, 128)
        self.fc3 = torch.nn.Linear(128, 2)  # Output (x, y) coordinates

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 25 * 12)  # Flatten the tensor
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Load the pre-trained model
model = EyeGazeCNN()
model.load_state_dict(torch.load('best_eye_gaze_model.pth', weights_only=True))
model.eval()  # Set the model to evaluation mode

# Set device to GPU or CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Define the transformation to match training
transform = transforms.Compose([
    transforms.Resize((50, 100)),  # Resize to match the CNN input
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Initialize MediaPipe Face Mesh
face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)

# Live eye capture
cam = cv.VideoCapture(0)

# Initialize the EMA smoothing variables
alpha = 0.7  # Adjust this to control smoothing (0 < alpha < 1)
prev_x, prev_y = None, None

def CaptureEye(frame_scale=0.15):
    ret, frame = cam.read()  # store single webcam frame

    rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    result = face_mesh.process(rgb_frame)

    if result.multi_face_landmarks:
        landmarks = result.multi_face_landmarks[0].landmark

        # Extract eye region using the same landmarks as training
        eye_box = frame[
            int(landmarks[27].y * frame.shape[0]):int(landmarks[23].y * frame.shape[0]),
            int(landmarks[226].x * frame.shape[1]):int(landmarks[190].x * frame.shape[1])
        ]

        eye_box = cv.cvtColor(eye_box, cv.COLOR_BGR2GRAY)  # Convert to grayscale
        eye_box = cv.resize(eye_box, dsize=(100, 50))  # Resize to match training

        # Convert the eye_box to a PIL image
        pil_image = Image.fromarray(eye_box)

        # Apply the transformation (resize, normalize, convert to tensor)
        eye_tensor = transform(pil_image).unsqueeze(0).to(device)  # Add batch dimension and move to device

        return eye_tensor, eye_box  # Return the preprocessed tensor and the raw eye_box for display

    else:
        print("No face detected.")
        return None, None

# Main loop for live model inference
while True:
    # Capture the eye image and preprocess it
    eye_tensor, eye_box = CaptureEye()

    if eye_tensor is not None:
        # Make predictions using the model
        with torch.no_grad():  # Disable gradients for inference
            outputs = model(eye_tensor)

        # Extract predicted (x, y) screen coordinates (raw values, no scaling required)
        predicted_coords = outputs.cpu().numpy()[0]
        x, y = predicted_coords[0], predicted_coords[1]

        # If it's the first prediction, initialize the previous coordinates
        if prev_x is None or prev_y is None:
            prev_x, prev_y = x, y

        # Apply Exponential Moving Average (EMA) for smoothing
        x = alpha * prev_x + (1 - alpha) * x
        y = alpha * prev_y + (1 - alpha) * y

        # Update previous coordinates
        prev_x, prev_y = x, y

        # Output predicted coordinates to the terminal
        print(f"Smoothed coordinates: x={x:.2f}, y={y:.2f}")

        # Move the mouse to the smoothed coordinates
        pyautogui.moveTo(int(x), int(y))

        # Display the captured eye box image
        cv.imshow('Eye Box', eye_box)

    # Exit if 'q' is pressed
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close OpenCV windows
cam.release()
cv.destroyAllWindows()
