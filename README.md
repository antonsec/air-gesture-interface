# Air Gesture Interface

A real-time **hand-controlled object playground** built with Python, [MediaPipe Hand Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker/python), and OpenCV. Use a webcam to pick up, move, rotate, resize, and throw virtual objects—no mouse required.

## Features

- **Either-hand grabbing:** pick up an object with your left or right hand by pinching your thumb and index finger together.
- **Two-hand manipulation:** the hand that grabbed an object moves/rotates it; the other hand changes its size by varying the thumb–index gap.
- **More reliable picking:** a visible pinch midpoint, an expanded grab region, rotation-aware hit testing, hand-relative pinch detection, and hysteresis to reduce gesture flicker.
- **Natural motion:** smoothing, predictive following, momentum when throwing, friction, bouncing, and gentle snapping near 0°, 90°, 180°, and 270°.
- **Flexible display:** a mirrored webcam feed fitted into a 960 × 600 canvas without intentional cropping, with a compact status HUD and adjustable window size.

## Requirements

- Python **3.9+**; the current dependencies were developed on an **Apple Silicon Mac with Python 3.9**.
- A working webcam and operating-system camera permission for the terminal or IDE used to start the app.
- Internet access for installing Python packages and downloading the Hand Landmarker model once.

The project uses the **MediaPipe Tasks API** (not the removed `mp.solutions.hands` API). The current `requirements.txt` pins **MediaPipe 1.1.1** and installs `opencv-contrib-python`. Installation and compatibility may differ on other operating systems and Python versions.

## Quick start (macOS / Linux)

Clone the repository:

```bash
git clone https://github.com/antonsec/air-gesture-interface.git
cd air-gesture-interface
```

Create a local Python environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install --no-compile -r requirements.txt
```

`--no-compile` is intentional: some MediaPipe 1.1.1 installations fail while compiling an invalid, bundled Python *test* file, even though the hand-tracking runtime can work. See [Troubleshooting](#troubleshooting).

Download Google's Hand Landmarker model:

```bash
mkdir -p models
curl -fL \
  "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task" \
  -o models/hand_landmarker.task
```

Run the app:

```bash
python main.py
```

The `.venv/` directory and downloaded `.task` model are local files and should **not** be committed to Git.

## Controls

| Gesture / key | Action |
| --- | --- |
| Left **or** right hand: thumb + index pinch over an object | Grab the object |
| Move the grabbing hand | Move the grabbed object |
| Twist the grabbing hand | Rotate the object |
| Change the **other** hand's thumb–index distance | Resize the object |
| Move quickly and release the grabbing hand's pinch | Throw the object |
| `R` | Reset object positions and sizes |
| `Space` | Pause / resume physics |
| `-` / `+` (or `=`) | Shrink / enlarge the display window |
| `Esc` | Exit |

The grab point is **between your thumb and index fingertips**, rather than at your index fingertip alone. Look for `L PINCH` or `R PINCH` near the hand cursor to confirm that a pinch is recognized. `OPEN` means the hand is tracked but the pinch isn't active. The hand that begins a grab keeps control of that object until release or tracking loss.

## Camera and display

The webcam image is mirrored to make interactions feel natural. The complete incoming frame is **fitted** within a 960 × 600 drawing area while retaining its aspect ratio. Dark borders are normal when the camera aspect ratio differs from the canvas. Resizing the window changes its displayed size, **not** the camera's field of view.

If the camera itself seems zoomed in, check the webcam's own auto-framing settings (for example, macOS Center Stage) and disable them. The app cannot recover parts of the scene that the camera or operating system has already cropped.

## Gesture tuning

The `Config` class in `main.py` contains useful thresholds:

| Setting | Purpose |
| --- | --- |
| `PINCH_CLOSE_RATIO` | Maximum hand-relative fingertip gap for starting a pinch |
| `PINCH_OPEN_RATIO` | Larger release threshold to avoid rapid grab/release flicker |
| `GRAB_PADDING` | Extra pixels around an object's rotated shape for easier selection |
| `RELEASE_FRAMES` | Consecutive open-hand frames before release |
| `LOST_TOLERANCE` | Frames to tolerate temporary hand-tracking loss |

Adjust these gradually if your lighting, camera angle, or hand size affects recognition.

## Troubleshooting

**`AttributeError: module 'mediapipe' has no attribute 'solutions'`**
You're running the legacy hand-tracking code with a newer MediaPipe package. Use this repository's updated `main.py` and `requirements.txt` together; the new code uses `mp.tasks.vision.HandLandmarker`.

**MediaPipe install fails on `text_embedder_test.py` with `invalid character '∂'`**
The error is in a file bundled with MediaPipe, not in this project. Retry the dependency installation with `python -m pip install --no-compile -r requirements.txt`. Confirm the result with `python -m pip check` and `python -c "import mediapipe as mp; print(mp.__version__)"`.

**Camera access is denied or the window never opens**
On macOS, enable camera access under **System Settings → Privacy & Security → Camera** for the app running Python (such as Terminal or VS Code). Restart the app after changing permissions; close other programs using the camera.

**`models/hand_landmarker.task` is missing**
Run the model download command from the repository root. The model is deliberately not tracked in Git.

**The fingers touch but nothing is grabbed**
Check the cursor label: it should show `PINCH`. Keep both fingertips visible, avoid strong backlighting, and move the midpoint onto an object. You can tune `PINCH_CLOSE_RATIO` and `GRAB_PADDING` in `Config`.

**Hand labels seem reversed**
The application mirrors the camera image and applies a handedness correction. Some camera drivers may already mirror input, so handedness should be checked on the actual device.

## How it works

`main.py` reads webcam frames with OpenCV and passes RGB images to MediaPipe's Hand Landmarker in `VIDEO` mode, using increasing timestamps. Landmark positions are smoothed and converted into hand gestures and cursor positions. The interaction layer assigns each grabbed object to one hand, while the other hand can change its size. OpenCV renders the objects, camera feed, cursors, and HUD; a lightweight physics loop handles free movement and collisions.

## Project structure

```text
air-gesture-interface/
├── main.py                 # Camera, gesture recognition, rendering, physics
├── requirements.txt        # Python dependencies
├── README.md               # Setup, controls, troubleshooting
├── .gitignore
└── models/
    └── hand_landmarker.task  # Download separately; not tracked in Git
```

## Status

This is an experimental webcam interface. The application has been migrated to MediaPipe Tasks and redesigned for two-hand interaction; gesture reliability still depends on lighting, camera position, and hardware. Test the grab, release, rotate, resize, and throw controls on your own device before merging changes into `main`.
