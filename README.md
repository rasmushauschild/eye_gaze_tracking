**Eye-Tracking with Deep Learning and Real-Time Mouse Control**
This project demonstrates real-time eye-tracking using a custom-trained convolutional neural network (CNN) to predict screen coordinates based on eye images captured from a webcam. The predicted gaze coordinates are used to control the mouse cursor on the screen.

**Features**
Real-time eye tracking: Uses a webcam feed to capture images of the eye in real time.
CNN-based prediction: A deep learning model (CNN) is trained to predict the (x, y) coordinates on the screen where the user is looking.
Mouse control: The predicted coordinates are used to move the mouse cursor on the screen.
Smoothing and responsiveness: Implements an exponential moving average (EMA) smoothing algorithm with a movement threshold for responsive yet stable cursor control.
