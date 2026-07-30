# Hand Object Manipulator

Real-time hand tracking object manipulation using MediaPipe + OpenCV.

Grab, move, rotate, scale, and throw virtual objects with your hands. Includes basic physics (velocity, friction, wall bouncing, gravity).

## Features

- **Right hand**: Pinch to grab & move, twist to rotate, fling to throw
- **Left hand**: Pinch in/out to scale/zoom the grabbed object
- Physics: momentum, friction, wall bounce, optional gravity
- Smooth adaptive tracking + predictive follow
- Clean neon-style visuals

## Requirements

- Python 3.8+
- Webcam

```bash
pip install -r requirements.txt