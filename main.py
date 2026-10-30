"""
Hand Object Manipulator
----------------------
Real-time hand tracking object manipulation with MediaPipe + OpenCV.
Supports grab, move, rotate, scale, throw, soft angle snap and basic physics.
"""

import cv2
import mediapipe as mp
import numpy as np
import time
import sys
from pathlib import Path
import math
import copy
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict


# ═══════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════

@dataclass
class Config:
    # Camera
    CAM_WIDTH: int = 960  # logical drawing canvas
    CAM_HEIGHT: int = 600

    # Tracking
    PINCH_THRESHOLD: float = 0.050  # retained for reference
    PINCH_CLOSE_RATIO: float = 0.48  # tip gap / palm size to start a pinch
    PINCH_OPEN_RATIO: float = 0.70   # larger threshold prevents flicker
    GRAB_PADDING: float = 18.0      # pixels outside visible object
    BASE_SMOOTH: float = 0.40
    FAST_SMOOTH: float = 0.75
    SPEED_REF: float = 25.0
    PREDICTION: float = 0.35
    HOLD_FOLLOW: float = 0.94

    # Physics
    FRICTION: float = 0.96
    ANGULAR_FRICTION: float = 0.93
    BOUNCE: float = 0.65
    THROW_MULTIPLIER: float = 1.2
    SPIN_MULTIPLIER: float = 0.8

    # Interaction
    LOST_TOLERANCE: int = 8
    RELEASE_FRAMES: int = 3
    MIN_SCALE: float = 0.55
    MAX_SCALE: float = 1.85

    # Soft angle snap
    SNAP_ZONE: float = 14.0
    SNAP_STRENGTH: float = 0.35


CFG = Config()


# ═══════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════

@dataclass
class Object2D:
    x: float
    y: float
    w: float
    h: float
    color: Tuple[int, int, int]
    label: str
    angle: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    va: float = 0.0

    @property
    def center(self) -> Tuple[float, float]:
        return self.x + self.w * 0.5, self.y + self.h * 0.5

    def contains(self, px: float, py: float, padding: float = 0.0) -> bool:
        """Rotate the cursor into object-local coordinates before hit testing."""
        cx, cy = self.center
        dx, dy = px - cx, py - cy
        theta = math.radians(self.angle)
        local_x = dx * math.cos(theta) + dy * math.sin(theta)
        local_y = -dx * math.sin(theta) + dy * math.cos(theta)
        return (abs(local_x) <= self.w * 0.5 + padding and
                abs(local_y) <= self.h * 0.5 + padding)

    def reset_velocity(self):
        self.vx = self.vy = self.va = 0.0


@dataclass
class HandData:
    index: Tuple[float, float]
    thumb: Tuple[float, float]
    velocity: Tuple[float, float]
    is_pinching: bool
    pinch_distance: float
    angle: float

    @property
    def pinch_point(self) -> Tuple[float, float]:
        """Grab from between the two fingertips, not the index tip alone."""
        return ((self.index[0] + self.thumb[0]) * 0.5,
                (self.index[1] + self.thumb[1]) * 0.5)


# ═══════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════

def normalize_angle(angle: float) -> float:
    return angle % 360.0


def soft_snap(angle: float) -> float:
    """Apply gentle magnetic pull toward 0° / 90° / 180° / 270°."""
    a = normalize_angle(angle)
    best_diff = None

    for target in (0.0, 90.0, 180.0, 270.0):
        diff = a - target
        if diff > 180:
            diff -= 360
        if diff < -180:
            diff += 360
        if abs(diff) <= CFG.SNAP_ZONE:
            if best_diff is None or abs(diff) < abs(best_diff):
                best_diff = diff

    if best_diff is not None:
        proximity = 1.0 - (abs(best_diff) / CFG.SNAP_ZONE)
        pull = best_diff * CFG.SNAP_STRENGTH * (0.4 + 0.6 * proximity)
        return angle - pull
    return angle


def adaptive_smooth(
    prev: Optional[Tuple[float, float]],
    raw: Tuple[float, float],
    prev_raw: Tuple[float, float]
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """Speed-adaptive exponential smoothing."""
    if prev is None:
        return raw, (0.0, 0.0)

    dx = raw[0] - prev_raw[0]
    dy = raw[1] - prev_raw[1]
    speed = math.hypot(dx, dy)
    t = min(1.0, speed / CFG.SPEED_REF)
    alpha = CFG.BASE_SMOOTH + (CFG.FAST_SMOOTH - CFG.BASE_SMOOTH) * t

    smoothed = (
        prev[0] * (1 - alpha) + raw[0] * alpha,
        prev[1] * (1 - alpha) + raw[1] * alpha
    )
    return smoothed, (dx, dy)


def get_pinch_distance(hand_landmarks) -> float:
    thumb = hand_landmarks[4]
    index = hand_landmarks[8]
    return math.hypot(thumb.x - index.x, thumb.y - index.y)


def get_hand_angle(hand_landmarks) -> float:
    wrist = hand_landmarks[0]
    middle = hand_landmarks[9]
    return math.degrees(math.atan2(middle.y - wrist.y, middle.x - wrist.x))


# ═══════════════════════════════════════════════════════════════
# PHYSICS
# ═══════════════════════════════════════════════════════════════

def update_physics(objects: List[Object2D], frame_w: int, frame_h: int, paused: bool, held_idx: Optional[int] = None):
    if paused:
        return

    for i, obj in enumerate(objects):
        if i == held_idx:
            continue  # keep the held object precisely under the hand
        obj.vx *= CFG.FRICTION
        obj.vy *= CFG.FRICTION
        obj.va *= CFG.ANGULAR_FRICTION

        obj.x += obj.vx
        obj.y += obj.vy
        obj.angle += obj.va

        # Wall collisions
        if obj.x < 0:
            obj.x = 0
            obj.vx *= -CFG.BOUNCE
        if obj.y < 0:
            obj.y = 0
            obj.vy *= -CFG.BOUNCE
        if obj.x + obj.w > frame_w:
            obj.x = frame_w - obj.w
            obj.vx *= -CFG.BOUNCE
        if obj.y + obj.h > frame_h:
            obj.y = frame_h - obj.h
            obj.vy *= -CFG.BOUNCE

    # Simple object-object separation
    n = len(objects)
    for i in range(n):
        for j in range(i + 1, n):
            if i == held_idx or j == held_idx:
                continue
            a, b = objects[i], objects[j]
            if not (a.x < b.x + b.w and a.x + a.w > b.x and
                    a.y < b.y + b.h and a.y + a.h > b.y):
                continue

            acx, acy = a.center
            bcx, bcy = b.center
            dx, dy = acx - bcx, acy - bcy
            dist = math.hypot(dx, dy) or 1.0
            overlap = 0.2 * (min(a.w, b.w) + min(a.h, b.h)) - dist

            if overlap > 0:
                nx, ny = dx / dist, dy / dist
                push = overlap * 0.5
                a.x += nx * push
                a.y += ny * push
                b.x -= nx * push
                b.y -= ny * push


def fit_camera_frame(frame: np.ndarray, canvas_width: int, canvas_height: int) -> np.ndarray:
    """Fit the entire camera image into a fixed canvas; never crop/stretch it."""
    source_height, source_width = frame.shape[:2]
    if source_width <= 0 or source_height <= 0:
        raise ValueError("Received an empty camera frame")

    ratio = min(canvas_width / source_width, canvas_height / source_height)
    fitted_width = max(1, int(source_width * ratio))
    fitted_height = max(1, int(source_height * ratio))
    interpolation = cv2.INTER_AREA if ratio < 1 else cv2.INTER_LINEAR
    fitted = cv2.resize(frame, (fitted_width, fitted_height), interpolation=interpolation)

    # Letterbox rather than center-cropping: hand and object coordinates use
    # this same canvas, so detection, rendering, and pointer hitboxes align.
    canvas = np.empty((canvas_height, canvas_width, 3), dtype=np.uint8)
    canvas[:] = (20, 24, 30)
    left = (canvas_width - fitted_width) // 2
    top = (canvas_height - fitted_height) // 2
    canvas[top:top + fitted_height, left:left + fitted_width] = fitted
    return canvas


# ═══════════════════════════════════════════════════════════════
# RENDERING
# ═══════════════════════════════════════════════════════════════

def draw_object(img: np.ndarray, obj: Object2D, active: bool = False, hover: bool = False):
    cx, cy = obj.center
    rad = math.radians(obj.angle)
    c, s = math.cos(rad), math.sin(rad)
    dx, dy = obj.w * 0.5, obj.h * 0.5

    corners = np.array([
        [cx + (-dx * c - -dy * s), cy + (-dx * s + -dy * c)],
        [cx + ( dx * c - -dy * s), cy + ( dx * s + -dy * c)],
        [cx + ( dx * c -  dy * s), cy + ( dx * s +  dy * c)],
        [cx + (-dx * c -  dy * s), cy + (-dx * s +  dy * c)],
    ], dtype=np.int32)

    # Glow (only when interacting)
    if active or hover:
        overlay = img.copy()
        cv2.polylines(overlay, [corners], True, obj.color, 7, cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.18, img, 0.82, 0, img)

    # Fill
    overlay = img.copy()
    cv2.fillPoly(overlay, [corners], obj.color)
    intensity = 0.22 if active else (0.14 if hover else 0.08)
    cv2.addWeighted(overlay, intensity, img, 1.0 - intensity, 0, img)

    # Border
    thickness = 3 if active else 2
    cv2.polylines(img, [corners], True, obj.color, thickness, cv2.LINE_AA)

    # Snap indicator
    if active:
        a = normalize_angle(obj.angle)
        for target in (0, 90, 180, 270):
            diff = min(abs(a - target), 360 - abs(a - target))
            if diff <= CFG.SNAP_ZONE:
                radius = int(16 + 10 * (1.0 - diff / CFG.SNAP_ZONE))
                cv2.circle(img, (int(cx), int(cy)), radius, (255, 255, 255), 1, cv2.LINE_AA)
                break

    # Label
    cv2.putText(img, obj.label, (int(cx) - 10, int(cy) + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)


def draw_hud(
    img: np.ndarray, status: str, fps: float, scale: float, angle: float,
    holding: bool, right_seen: bool, left_seen: bool, paused: bool
):
    """Low-profile translucent HUD, sized for a 960 x 600 canvas."""
    h, w = img.shape[:2]
    layer = img.copy()
    cv2.rectangle(layer, (0, 0), (w, 76), (17, 21, 27), -1)
    cv2.rectangle(layer, (0, h - 48), (w, h), (17, 21, 27), -1)
    cv2.addWeighted(layer, 0.84, img, 0.16, 0, img)
    accent = (180, 245, 95)  # BGR neon-lime accent
    muted = (174, 183, 193)
    white = (241, 246, 250)
    cv2.line(img, (20, 76), (w - 20, 76), (45, 61, 67), 1, cv2.LINE_AA)

    cv2.putText(img, "AIR / GESTURE", (22, 31), cv2.FONT_HERSHEY_SIMPLEX,
                0.69, white, 2, cv2.LINE_AA)
    cv2.putText(img, "INTERACTIVE OBJECT LAB", (23, 56),
                cv2.FONT_HERSHEY_SIMPLEX, 0.36, muted, 1, cv2.LINE_AA)

    # Detector badges; green means a hand is currently tracked.
    for x, name, seen in ((476, "RIGHT", right_seen), (600, "LEFT", left_seen)):
        cv2.circle(img, (x, 27), 5, accent if seen else (87, 94, 101), -1, cv2.LINE_AA)
        cv2.putText(img, name, (x + 13, 32), cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, white if seen else muted, 1, cv2.LINE_AA)

    status_color = (65, 190, 255) if paused else (90, 235, 170) if holding else muted
    cv2.putText(img, status, (743, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.52, status_color, 1, cv2.LINE_AA)
    cv2.putText(img, "{:02d} FPS".format(min(int(fps), 99)), (857, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.41, muted, 1, cv2.LINE_AA)
    if holding:
        cv2.putText(img, "SCALE {:3d}%   ANGLE {:3d} deg".format(
            int(scale * 100), int(angle % 360)),
            (22, 104), cv2.FONT_HERSHEY_SIMPLEX, 0.40, accent, 1, cv2.LINE_AA)

    cv2.putText(img, "EITHER HAND: GRAB / MOVE    OTHER HAND: RESIZE",
                (22, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.39,
                white, 1, cv2.LINE_AA)
    cv2.putText(img, "[ - / + ] WINDOW   [ R ] RESET   [ SPACE ] PAUSE   [ ESC ] EXIT",
                (508, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.33,
                muted, 1, cv2.LINE_AA)


# ═══════════════════════════════════════════════════════════════
# HAND TRACKER
# ═══════════════════════════════════════════════════════════════

class HandTracker:
    """Two-hand tracker powered by the MediaPipe Tasks Hand Landmarker."""

    def __init__(self):
        model_path = Path(__file__).resolve().parent / "models" / "hand_landmarker.task"
        if not model_path.is_file():
            raise FileNotFoundError(
                f"Hand Landmarker model not found: {model_path}\n"
                "See README.md to download models/hand_landmarker.task."
            )

        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.45,
            min_hand_presence_confidence=0.45,
            min_tracking_confidence=0.45,
        )
        self.hands = mp.tasks.vision.HandLandmarker.create_from_options(options)
        self.smooth_state: Dict[str, dict] = {}
        self.last_timestamp_ms = -1

    def process(self, frame: np.ndarray) -> Tuple[Optional[HandData], Optional[HandData]]:
        h, w = frame.shape[:2]
        rgb = np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # VIDEO mode requires strictly increasing timestamps, in milliseconds.
        timestamp_ms = max(time.monotonic_ns() // 1_000_000, self.last_timestamp_ms + 1)
        self.last_timestamp_ms = timestamp_ms
        results = self.hands.detect_for_video(mp_image, timestamp_ms)

        right = left = None
        if not results.hand_landmarks or not results.handedness:
            self.smooth_state.clear()
            return None, None

        for landmarks, handedness in zip(results.hand_landmarks, results.handedness):
            if not handedness:
                continue
            # The main loop mirrors the camera input (cv2.flip(frame, 1)).
            # Invert the Task API's labels to identify the user's physical hands.
            detected_label = handedness[0].category_name
            if detected_label not in ("Left", "Right"):
                continue
            label = "Right" if detected_label == "Left" else "Left"

            raw_index = (landmarks[8].x * w, landmarks[8].y * h)
            raw_thumb = (landmarks[4].x * w, landmarks[4].y * h)
            if label not in self.smooth_state:
                self.smooth_state[label] = {
                    "pt": raw_index,
                    "prev": raw_index,
                    "vel": (0.0, 0.0),
                    "thumb": raw_thumb,
                    "pinching": False,
                }
            else:
                state = self.smooth_state[label]
                new_pt, vel = adaptive_smooth(state["pt"], raw_index, state["prev"])
                state["prev"] = state["pt"]
                state["pt"] = new_pt
                state["vel"] = vel
                state["thumb"] = (
                    state["thumb"][0] * 0.55 + raw_thumb[0] * 0.45,
                    state["thumb"][1] * 0.55 + raw_thumb[1] * 0.45,
                )

            state = self.smooth_state[label]
            # Use the *displayed smoothed fingertips* for pinch classification.
            # Normalize to the visible palm size, not the entire camera frame.
            palm_width = math.hypot((landmarks[5].x - landmarks[17].x) * w,
                                    (landmarks[5].y - landmarks[17].y) * h)
            palm_length = math.hypot((landmarks[0].x - landmarks[9].x) * w,
                                     (landmarks[0].y - landmarks[9].y) * h)
            palm_size = max(20.0, 0.7 * palm_width + 0.3 * palm_length)
            fingertip_gap = math.hypot(state["pt"][0] - state["thumb"][0],
                                       state["pt"][1] - state["thumb"][1])
            gap_ratio = fingertip_gap / palm_size
            cutoff = (CFG.PINCH_OPEN_RATIO if state.get("pinching", False)
                      else CFG.PINCH_CLOSE_RATIO)
            state["pinching"] = gap_ratio <= cutoff
            # Preserve normalized gap for the other-hand size control.
            pinch_distance = get_pinch_distance(landmarks)
            data = HandData(
                index=state["pt"],
                thumb=state["thumb"],
                velocity=state["vel"],
                is_pinching=state["pinching"],
                pinch_distance=pinch_distance,
                angle=get_hand_angle(landmarks),
            )
            if label == "Right":
                right = data
            else:
                left = data
        return right, left

    def draw_cursors(self, frame: np.ndarray, right: Optional[HandData], left: Optional[HandData]):
        """Show each finger and the precise grab hotspot, plus pinch state."""
        for label, hand in (("R", right), ("L", left)):
            if hand is None:
                continue
            color = (100, 245, 155) if hand.is_pinching else (200, 125, 245)
            ix, iy = (int(v) for v in hand.index)
            tx, ty = (int(v) for v in hand.thumb)
            px, py = (int(v) for v in hand.pinch_point)
            cv2.line(frame, (ix, iy), (tx, ty), color, 1, cv2.LINE_AA)
            cv2.circle(frame, (ix, iy), 6, color, -1, cv2.LINE_AA)
            cv2.circle(frame, (tx, ty), 6, color, -1, cv2.LINE_AA)
            cv2.circle(frame, (px, py), 9 if hand.is_pinching else 5,
                       color, 2, cv2.LINE_AA)
            cv2.putText(frame, f"{label} {'PINCH' if hand.is_pinching else 'OPEN'}",
                        (px + 13, py - 12), cv2.FONT_HERSHEY_SIMPLEX,
                        0.38, color, 1, cv2.LINE_AA)


    def close(self):
        self.hands.close()

# ═══════════════════════════════════════════════════════════════
# MAIN APPLICATION
# ═══════════════════════════════════════════════════════════════

class ObjectManipulator:
    def __init__(self):
        # Keep the webcam's native aspect ratio instead of forcing 800x450,
        # which can make some cameras crop or digitally zoom their picture.
        backend = cv2.CAP_AVFOUNDATION if sys.platform == "darwin" else cv2.CAP_ANY
        self.cam = cv2.VideoCapture(0, backend)
        if not self.cam.isOpened():
            raise RuntimeError(
                "Webcam unavailable. Allow camera access in your OS privacy "
                "settings and close other apps using the camera."
            )
        self.cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.window_name = "AIR GESTURE | Object Lab"
        self.window_scale = 0.82
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        self.resize_display()

        self.tracker = HandTracker()

        self.objects: List[Object2D] = [
            Object2D(155, 175, 132, 92, (75, 227, 255), "01"),
            Object2D(425, 180, 112, 112, (255, 120, 221), "02"),
            Object2D(700, 310, 138, 90, (125, 239, 150), "03"),
        ]
        self.original_objects = copy.deepcopy(self.objects)

        # Interaction state
        self.grabbed_idx: Optional[int] = None
        self.grab_hand: Optional[str] = None
        self.offset = (0.0, 0.0)
        self.base_left_pinch: Optional[float] = None
        self.base_size: Optional[Tuple[float, float]] = None
        self.smooth_scale = 1.0
        self.prev_angle: Optional[float] = None
        self.lost_counter = 0
        self.open_counter = 0
        self.hover_idx: Optional[int] = None
        self.physics_paused = False

        self.prev_time = time.perf_counter()

    def resize_display(self):
        """Resize the *window*, not the tracking canvas or camera crop."""
        width = max(480, int(CFG.CAM_WIDTH * self.window_scale))
        height = max(300, int(CFG.CAM_HEIGHT * self.window_scale))
        cv2.resizeWindow(self.window_name, width, height)

    def _start_grab(self, idx: int, hand: HandData, left: Optional[HandData], hand_label: str):
        obj = self.objects[idx]
        self.grabbed_idx = idx
        self.grab_hand = hand_label
        self.offset = (hand.pinch_point[0] - obj.x, hand.pinch_point[1] - obj.y)
        self.base_size = (obj.w, obj.h)
        self.smooth_scale = 1.0
        self.base_left_pinch = left.pinch_distance if left else None
        self.open_counter = 0
        self.prev_angle = hand.angle
        obj.reset_velocity()

        # Bring to front
        self.objects.append(self.objects.pop(idx))
        self.grabbed_idx = len(self.objects) - 1

    def _release(self, hand: HandData):
        obj = self.objects[self.grabbed_idx]
        px, py = hand.pinch_point
        vx, vy = hand.velocity

        obj.x = px - self.offset[0]
        obj.y = py - self.offset[1]
        obj.angle = soft_snap(obj.angle)
        obj.vx = vx * CFG.THROW_MULTIPLIER
        obj.vy = vy * CFG.THROW_MULTIPLIER

        if self.prev_angle is not None:
            delta = hand.angle - self.prev_angle
            if delta > 180:
                delta -= 360
            if delta < -180:
                delta += 360
            obj.va = delta * CFG.SPIN_MULTIPLIER

        self.grabbed_idx = None
        self.grab_hand = None
        self.base_left_pinch = None
        self.base_size = None
        self.prev_angle = None
        self.open_counter = 0

    def _update_held_object(self, hand: HandData, left: Optional[HandData]):
        obj = self.objects[self.grabbed_idx]
        px, py = hand.pinch_point
        vx, vy = hand.velocity

        # Predictive tight follow
        pred_x = px + vx * CFG.PREDICTION
        pred_y = py + vy * CFG.PREDICTION
        target_x = pred_x - self.offset[0]
        target_y = pred_y - self.offset[1]

        obj.x = obj.x * (1 - CFG.HOLD_FOLLOW) + target_x * CFG.HOLD_FOLLOW
        obj.y = obj.y * (1 - CFG.HOLD_FOLLOW) + target_y * CFG.HOLD_FOLLOW
        obj.reset_velocity()

        # Rotation + soft snap
        if self.prev_angle is not None:
            delta = hand.angle - self.prev_angle
            if delta > 180:
                delta -= 360
            if delta < -180:
                delta += 360
            obj.angle = soft_snap(obj.angle + delta * 0.96)
        self.prev_angle = hand.angle

        # Left-hand zoom
        if left is not None:
            if self.base_left_pinch is None or self.base_left_pinch < 0.015:
                self.base_left_pinch = max(left.pinch_distance, 0.02)
                self.base_size = (obj.w, obj.h)
                self.smooth_scale = 1.0
            else:
                target = left.pinch_distance / self.base_left_pinch
                target = max(CFG.MIN_SCALE, min(CFG.MAX_SCALE, target))
                self.smooth_scale = self.smooth_scale * 0.8 + target * 0.2

                new_w = self.base_size[0] * self.smooth_scale
                new_h = self.base_size[1] * self.smooth_scale
                cx, cy = obj.center
                obj.w, obj.h = new_w, new_h
                obj.x = cx - new_w * 0.5
                obj.y = cy - new_h * 0.5
        else:
            self.base_left_pinch = None
            self.base_size = (obj.w, obj.h)
            self.smooth_scale = 1.0

    def handle_interaction(self, right: Optional[HandData], left: Optional[HandData]):
        """Either hand may grab; the opposite hand adjusts size while held."""
        self.hover_idx = None
        hands = {"Right": right, "Left": left}

        if self.grabbed_idx is None:
            # Search from front to back (highest index is drawn last).
            # Both fingertip dots move around the actual pinch point.
            # Keep a small hitbox tolerance, including for rotated objects.
            for label, hand in hands.items():
                if hand is None:
                    continue
                px, py = hand.pinch_point
                for i in range(len(self.objects) - 1, -1, -1):
                    if not self.objects[i].contains(px, py, CFG.GRAB_PADDING):
                        continue
                    if self.hover_idx is None:
                        self.hover_idx = i
                    if hand.is_pinching:
                        other = hands["Left" if label == "Right" else "Right"]
                        self._start_grab(i, hand, other, label)
                        return
                    break
            return

        # Lock ownership to the hand that initiated the grab; switching hands
        # silently would make an object teleport when one hand disappears.
        owner = hands.get(self.grab_hand)
        other = hands.get("Left" if self.grab_hand == "Right" else "Right")
        if owner is None:
            self.lost_counter += 1
            if self.lost_counter > CFG.LOST_TOLERANCE:
                self.objects[self.grabbed_idx].angle = soft_snap(
                    self.objects[self.grabbed_idx].angle
                )
                self._clear_grab()
            return

        self.lost_counter = 0
        if owner.is_pinching:
            self.open_counter = 0
        else:
            self.open_counter += 1

        if self.open_counter >= CFG.RELEASE_FRAMES:
            self._release(owner)
        else:
            self._update_held_object(owner, other)

    def _clear_grab(self):
        self.grabbed_idx = None
        self.grab_hand = None
        self.base_left_pinch = None
        self.base_size = None
        self.prev_angle = None
        self.open_counter = 0
        self.lost_counter = 0
        self.smooth_scale = 1.0

    def run(self):
        while True:
            if not self.cam.grab():
                break
            ok, frame = self.cam.retrieve()
            if not ok:
                break

            frame = fit_camera_frame(cv2.flip(frame, 1), CFG.CAM_WIDTH, CFG.CAM_HEIGHT)
            h, w = frame.shape[:2]

            # Tracking
            right, left = self.tracker.process(frame)

            # Interaction
            self.handle_interaction(right, left)

            # Physics
            update_physics(self.objects, w, h, self.physics_paused, self.grabbed_idx)

            # Render
            frame = cv2.addWeighted(frame, 0.87, np.zeros_like(frame), 0.13, 0)

            for i, obj in enumerate(self.objects):
                draw_object(
                    frame, obj,
                    active=(i == self.grabbed_idx),
                    hover=(i == self.hover_idx)
                )

            # Connection line while holding
            if self.grabbed_idx is not None:
                owner = right if self.grab_hand == "Right" else left
                if owner is not None:
                    obj = self.objects[self.grabbed_idx]
                    cx, cy = obj.center
                    cv2.line(frame,
                         (int(owner.pinch_point[0]), int(owner.pinch_point[1])),
                         (int(cx), int(cy)),
                         (0, 255, 220), 1, cv2.LINE_AA)

            self.tracker.draw_cursors(frame, right, left)
            # HUD
            now = time.perf_counter()
            fps = 1.0 / max(now - self.prev_time, 1e-6)
            self.prev_time = now

            status = "PAUSED" if self.physics_paused else ("HOLDING" if self.grabbed_idx is not None else "READY")
            angle = self.objects[self.grabbed_idx].angle if self.grabbed_idx is not None else 0
            draw_hud(frame, status, fps, self.smooth_scale, angle,
                     self.grabbed_idx is not None, right is not None, left is not None, self.physics_paused)

            cv2.imshow(self.window_name, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:                       # ESC
                break
            if key in (ord("r"), ord("R")):     # Reset
                self.objects = copy.deepcopy(self.original_objects)
                self.grabbed_idx = None
                self.grab_hand = None
                self.base_left_pinch = None
                self.base_size = None
                self.prev_angle = None
                self.smooth_scale = 1.0
            if key == 32:                       # Space – pause physics
                self.physics_paused = not self.physics_paused
            if key in (ord("-"), ord("_")):
                self.window_scale = max(0.55, round(self.window_scale - 0.1, 2))
                self.resize_display()
            if key in (ord("+"), ord("=")):
                self.window_scale = min(1.4, round(self.window_scale + 0.1, 2))
                self.resize_display()

        self.cleanup()

    def cleanup(self):
        self.cam.release()
        self.tracker.close()
        cv2.destroyAllWindows()


# ═══════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    app = ObjectManipulator()
    app.run()