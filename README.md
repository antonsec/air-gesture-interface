# Hand Object Manipulator

Real-time hand-tracking object manipulation built with **MediaPipe** + **OpenCV**.

Grab, move, rotate, scale and throw virtual objects using your hands. Includes soft angle snapping and lightweight physics.

---

## Features

- **Right hand**
  - Pinch → grab & move
  - Twist hand → rotate object
  - Fling + release → throw with momentum
- **Left hand**
  - Pinch in/out → scale / zoom object
- Soft magnetic snap to 0° / 90° / 180° / 270°
- Basic physics (velocity, friction, wall bounce)
- Adaptive smoothing + predictive tracking
- Clean modular codebase

---

## Requirements

- Python 3.8+
- Webcam

```bash
pip install -r requirements.txt