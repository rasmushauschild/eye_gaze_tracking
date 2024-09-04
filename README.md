# Eye-Tracking with Deep Learning and Real-Time Mouse Control

This project controls the cursor on the screen with real-time eye-tracking using a custom-trained convolutional neural network (CNN) based on eye images captured from a webcam.

## Features
- **Real-time eye tracking**: Uses a webcam feed to capture images of the eye in real time.
- **CNN-based prediction**: A CNN is trained to predict the (x, y) coordinates on the screen where the user is looking.
- **Mouse control**: The predicted coordinates are used to move the mouse cursor on the screen.
- **Smoothing and responsiveness**: Implements an exponential moving average (EMA) smoothing algorithm with a movement threshold for responsive yet (somewhat) stable cursor control.
