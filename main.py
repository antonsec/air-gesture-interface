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
import itertools
from dataclasses import dataclass
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
    TRACKING_WIDTH: int = 640  # inference only; display keeps its resolution
    PINCH_CLOSE_RATIO: float = 0.25  # tip gap / palm size to start a pinch
    PINCH_OPEN_RATIO: float = 0.40   # larger threshold prevents flicker
    GRAB_PADDING: float = 18.0      # pixels outside visible object
    BASE_SMOOTH: float = 0.60
    FAST_SMOOTH: float = 0.85
    SPEED_REF: float = 25.0
    PREDICTION: float = 0.35
    HOLD_FOLLOW: float = 0.94

    # Physics
    FRICTION: float = 0.96          # retained velocity per 1/30 second
    ANGULAR_FRICTION: float = 0.93
    BOUNCE: float = 0.65
    THROW_MULTIPLIER: float = 1.2
    SPIN_MULTIPLIER: float = 0.8

    # Interaction
    LOST_SECONDS: float = 0.30
    RELEASE_SECONDS: float = 0.10
    REFERENCE_FPS: float = 30.0
    MAX_PHYSICS_DT: float = 0.10
    PHYSICS_STEP: float = 1.0 / 120.0
    MATCH_DISTANCE: float = 180.0
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

    def axes(self):
        theta = math.radians(self.angle)
        c, s = math.cos(theta), math.sin(theta)
        return (c, s), (-s, c)

    def corners(self):
        cx, cy = self.center
        u, v = self.axes()
        return [(cx + sx * self.w * 0.5 * u[0] + sy * self.h * 0.5 * v[0],
                 cy + sx * self.w * 0.5 * u[1] + sy * self.h * 0.5 * v[1])
                for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))]

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
    gap_ratio: float = 0.0
    reacquired: bool = False

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


def soft_snap(angle: float, dt: float = 1.0 / 30.0) -> float:
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
        strength = CFG.SNAP_STRENGTH * (0.4 + 0.6 * proximity)
        pull = 1 - (1 - strength) ** (dt * CFG.REFERENCE_FPS)
        return angle - best_diff * pull
    return angle


def adaptive_smooth(
    prev: Optional[Tuple[float, float]],
    raw: Tuple[float, float],
    prev_raw: Tuple[float, float],
    dt: float = 1.0 / 30.0
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """Speed-adaptive exponential smoothing."""
    if prev is None:
        return raw, (0.0, 0.0)

    dx = raw[0] - prev_raw[0]
    dy = raw[1] - prev_raw[1]
    speed = math.hypot(dx, dy) / max(dt * CFG.REFERENCE_FPS, 1e-6)
    t = min(1.0, speed / CFG.SPEED_REF)
    alpha = CFG.BASE_SMOOTH + (CFG.FAST_SMOOTH - CFG.BASE_SMOOTH) * t

    alpha = 1 - (1 - alpha) ** (dt * CFG.REFERENCE_FPS)
    smoothed = (
        prev[0] * (1 - alpha) + raw[0] * alpha,
        prev[1] * (1 - alpha) + raw[1] * alpha
    )
    return smoothed, (dx, dy)


def get_pinch_distance(hand_landmarks) -> float:
    thumb = hand_landmarks[4]
    index = hand_landmarks[8]
    return math.hypot(thumb.x - index.x, thumb.y - index.y)


def get_hand_angle(hand_landmarks, width: int, height: int) -> float:
    wrist = hand_landmarks[0]
    middle = hand_landmarks[9]
    return math.degrees(math.atan2((middle.y - wrist.y) * height, (middle.x - wrist.x) * width))


# ═══════════════════════════════════════════════════════════════
# PHYSICS
# ═══════════════════════════════════════════════════════════════

def constrain_to_canvas(obj: Object2D, width: int, height: int):
    """Constrain rotated extents, reflecting only outward velocity."""
    u, v = obj.axes()
    extent_x = abs(u[0]) * obj.w * 0.5 + abs(v[0]) * obj.h * 0.5
    extent_y = abs(u[1]) * obj.w * 0.5 + abs(v[1]) * obj.h * 0.5
    cx, cy = obj.center
    for coordinate, extent, limit, position, velocity in (
        (cx, extent_x, width, "x", "vx"),
        (cy, extent_y, height, "y", "vy"),
    ):
        if extent * 2 > limit:
            target = limit * 0.5
            setattr(obj, velocity, 0.0)
        else:
            target = max(extent, min(limit - extent, coordinate))
            speed = getattr(obj, velocity)
            if (coordinate < extent and speed < 0) or (coordinate > limit - extent and speed > 0):
                setattr(obj, velocity, -speed * CFG.BOUNCE)
        setattr(obj, position, getattr(obj, position) + target - coordinate)


def collision_overlap(a: Object2D, b: Object2D):
    """Separating-axis test, returning the smallest separation from a to b."""
    acx, acy = a.center
    bcx, bcy = b.center
    best = None
    for nx, ny in (*a.axes(), *b.axes()):
        projections_a = [x * nx + y * ny for x, y in a.corners()]
        projections_b = [x * nx + y * ny for x, y in b.corners()]
        # Include full containment: interval intersection length is insufficient.
        overlap = min(max(projections_a) - min(projections_b),
                      max(projections_b) - min(projections_a))
        if overlap <= 0:
            return None
        if (bcx - acx) * nx + (bcy - acy) * ny < 0:
            nx, ny = -nx, -ny
        if best is None or overlap < best[2]:
            best = (nx, ny, overlap)
    return best


def update_physics(objects: List[Object2D], frame_w: int, frame_h: int,
                   paused: bool, held_idx: Optional[int] = None,
                   dt: float = 1.0 / 30.0):
    """Velocities are pixels/second and degrees/second. Bound long frame stalls."""
    if paused or dt <= 0:
        return
    remaining = min(dt, CFG.MAX_PHYSICS_DT)
    while remaining > 1e-9:
        step = min(CFG.PHYSICS_STEP, remaining)
        remaining -= step
        for i, obj in enumerate(objects):
            if i == held_idx:
                continue
            for position, velocity, friction in (
                ("x", "vx", CFG.FRICTION), ("y", "vy", CFG.FRICTION),
                ("angle", "va", CFG.ANGULAR_FRICTION),
            ):
                decay_rate = -math.log(friction) * CFG.REFERENCE_FPS
                decay = math.exp(-decay_rate * step)
                speed = getattr(obj, velocity)
                travel = speed * step if decay_rate == 0 else speed * (1 - decay) / decay_rate
                setattr(obj, position, getattr(obj, position) + travel)
                setattr(obj, velocity, speed * decay)
            constrain_to_canvas(obj, frame_w, frame_h)

        for i, a in enumerate(objects):
            for j in range(i + 1, len(objects)):
                b = objects[j]
                overlap = collision_overlap(a, b)
                if overlap is None:
                    continue
                nx, ny, depth = overlap
                weight_a, weight_b = float(i != held_idx), float(j != held_idx)
                total = weight_a + weight_b
                a.x -= nx * depth * weight_a / total
                a.y -= ny * depth * weight_a / total
                b.x += nx * depth * weight_b / total
                b.y += ny * depth * weight_b / total
                relative_speed = (b.vx - a.vx) * nx + (b.vy - a.vy) * ny
                if relative_speed < 0:
                    impulse = -(1 + CFG.BOUNCE) * relative_speed / total
                    a.vx -= impulse * nx * weight_a
                    a.vy -= impulse * ny * weight_a
                    b.vx += impulse * nx * weight_b
                    b.vy += impulse * ny * weight_b
        for obj in objects:
            constrain_to_canvas(obj, frame_w, frame_h)


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
    corners = np.array(obj.corners(), dtype=np.int32)

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

class PinchCalibration:
    """Session-only, per-hand thresholds from two deliberate one-second samples."""

    def __init__(self):
        self.thresholds = {}
        self.phase = None
        self.label = None
        self.samples = []
        self.elapsed = 0.0
        self.waited = 0.0
        self.open_ratio = None
        self.message = "C: calibrate pinch   X: restore defaults"

    @property
    def active(self):
        return self.phase is not None

    def reset(self):
        self.__init__()

    def advance(self, hands):
        if self.phase == "ready_pinch":
            if hands.get(self.label) is None:
                self.message = "Show the same hand, then press C"
                return
            self.phase = "pinch"
            self.message = "Keep thumb and index touching"
        elif self.phase is None:
            visible = [label for label, hand in hands.items() if hand is not None]
            if len(visible) != 1:
                self.message = "Show ONE open hand, then press C"
                return
            self.label = visible[0]
            self.phase = "open"
            self.message = "Keep thumb and index comfortably apart"
        else:
            return
        self.samples = []
        self.elapsed = self.waited = 0.0

    def update(self, hands, dt):
        if self.phase not in ("open", "pinch"):
            return
        self.waited += dt
        hand = hands.get(self.label)
        if hand is not None and not hand.reacquired:
            self.samples.append(hand.gap_ratio)
            self.elapsed += dt
        if self.waited > 10.0:
            self.phase = None
            self.message = "Calibration timed out. Show one hand and press C to retry"
            return
        if self.elapsed < 1.0 or len(self.samples) < 10:
            return
        ratio = float(np.median(self.samples))
        if self.phase == "open":
            self.open_ratio = ratio
            self.phase = "ready_pinch"
            self.message = "Touch thumb and index, then press C again"
            return
        if self.open_ratio < 0.45 or ratio > 0.45 or self.open_ratio - ratio < 0.25:
            self.message = "Poses too similar or pinch too wide. Press C to retry"
        else:
            gap = self.open_ratio - ratio
            close = max(0.08, min(0.40, ratio + gap * 0.20))
            release = max(close + 0.08, min(0.65, ratio + gap * 0.40))
            self.thresholds[self.label] = (close, release)
            self.message = f"{self.label} calibrated. C: calibrate again   X: defaults"
        self.phase = None


def assign_hand_labels(detections, states, now):
    """Associate wrists spatially, using handedness only as a tie-breaker."""
    labels = ("Right", "Left")
    best = None
    for assignment in itertools.permutations(labels, len(detections)):
        matches, cost = 0, 0.0
        for detection, label in zip(detections, assignment):
            state = states.get(label)
            cost += 5.0 * (label != detection["label"])
            if state is None:
                continue
            elapsed = min(now - state["last_seen"], 0.10)
            velocity = state["wrist_velocity"]
            predicted = (state["wrist"][0] + velocity[0] * elapsed,
                         state["wrist"][1] + velocity[1] * elapsed)
            distance = math.dist(detection["wrist"], predicted)
            if distance <= CFG.MATCH_DISTANCE:
                matches += 1
                cost += distance
            else:
                cost += 1000.0
        score = (-matches, cost)
        if best is None or score < best[0]:
            best = (score, assignment)
    return best[1] if best else ()


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
        self.calibration = PinchCalibration()

    def process(self, frame: np.ndarray) -> Tuple[Optional[HandData], Optional[HandData]]:
        h, w = frame.shape[:2]
        # Keep display quality separate from the cost of synchronous inference.
        tracking_frame = frame
        if w > CFG.TRACKING_WIDTH:
            tracking_frame = cv2.resize(
                frame, (CFG.TRACKING_WIDTH, max(1, round(h * CFG.TRACKING_WIDTH / w))),
                interpolation=cv2.INTER_AREA,
            )
        rgb = np.ascontiguousarray(cv2.cvtColor(tracking_frame, cv2.COLOR_BGR2RGB))
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # VIDEO mode requires strictly increasing timestamps, in milliseconds.
        timestamp_ms = max(time.monotonic_ns() // 1_000_000, self.last_timestamp_ms + 1)
        self.last_timestamp_ms = timestamp_ms
        results = self.hands.detect_for_video(mp_image, timestamp_ms)

        now = timestamp_ms / 1000.0
        expired_labels = {label for label, state in self.smooth_state.items()
                          if now - state["last_seen"] > CFG.LOST_SECONDS}
        self.smooth_state = {label: state for label, state in self.smooth_state.items()
                             if label not in expired_labels}
        detections = []
        for landmarks, handedness in zip(results.hand_landmarks or [], results.handedness or []):
            if not handedness or handedness[0].category_name not in ("Left", "Right"):
                continue
            # The camera image is mirrored before inference.
            label = "Right" if handedness[0].category_name == "Left" else "Left"
            detections.append({"label": label, "landmarks": landmarks,
                               "wrist": (landmarks[0].x * w, landmarks[0].y * h)})
        labels = assign_hand_labels(detections, self.smooth_state, now)
        right = left = None
        seen = set()
        for detection, label in zip(detections, labels):
            landmarks = detection["landmarks"]
            seen.add(label)
            raw_index = (landmarks[8].x * w, landmarks[8].y * h)
            raw_thumb = (landmarks[4].x * w, landmarks[4].y * h)
            previous = self.smooth_state.get(label)
            reacquired = label in expired_labels or (previous is not None and not previous["visible"])
            if previous is not None and math.dist(detection["wrist"], previous["wrist"]) > CFG.MATCH_DISTANCE:
                # A spatial discontinuity is not safe evidence of continuity.
                reacquired = True
            dt = max(0.001, now - previous["last_seen"]) if previous else 1 / CFG.REFERENCE_FPS
            if reacquired:
                self.smooth_state.pop(label, None)
            if label not in self.smooth_state:
                self.smooth_state[label] = {
                    "pt": raw_index,
                    "prev": raw_index,
                    "vel": (0.0, 0.0),
                    "thumb": raw_thumb,
                    "prev_thumb": raw_thumb,
                    "pinching": False,
                }
            else:
                state = self.smooth_state[label]
                new_pt, _ = adaptive_smooth(state["pt"], raw_index, state["prev"], dt)
                previous_midpoint = (
                    (state["pt"][0] + state["thumb"][0]) * 0.5,
                    (state["pt"][1] + state["thumb"][1]) * 0.5,
                )
                state["prev"] = raw_index
                state["pt"] = new_pt
                # Throw and prediction follow the same visible grab midpoint.
                state["thumb"], _ = adaptive_smooth(
                    state["thumb"], raw_thumb, state["prev_thumb"], dt
                )
                state["prev_thumb"] = raw_thumb

                state["vel"] = (
                    ((new_pt[0] + state["thumb"][0]) * 0.5 - previous_midpoint[0]) / dt,
                    ((new_pt[1] + state["thumb"][1]) * 0.5 - previous_midpoint[1]) / dt,
                )

            state = self.smooth_state[label]
            if previous is not None and not reacquired:
                state["wrist_velocity"] = (
                    (detection["wrist"][0] - previous["wrist"][0]) / dt,
                    (detection["wrist"][1] - previous["wrist"][1]) / dt,
                )
            else:
                state["wrist_velocity"] = (0.0, 0.0)
            state["last_seen"] = now
            state["wrist"] = detection["wrist"]
            state["visible"] = True
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
            close, release = self.calibration.thresholds.get(
                label, (CFG.PINCH_CLOSE_RATIO, CFG.PINCH_OPEN_RATIO))
            cutoff = release if state["pinching"] else close
            state["pinching"] = gap_ratio <= cutoff
            # Preserve normalized gap for the other-hand size control.
            pinch_distance = get_pinch_distance(landmarks)
            data = HandData(
                index=state["pt"],
                thumb=state["thumb"],
                velocity=state["vel"],
                is_pinching=state["pinching"],
                pinch_distance=pinch_distance,
                angle=get_hand_angle(landmarks, w, h),
                gap_ratio=gap_ratio,
                reacquired=reacquired,
            )
            if label == "Right":
                right = data
            else:
                left = data
        for label, state in self.smooth_state.items():
            state["visible"] = label in seen
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

        try:
            self.tracker = HandTracker()
        except Exception:
            self.cam.release()
            cv2.destroyAllWindows()
            raise

        self.objects: List[Object2D] = [
            Object2D(155, 175, 132, 92, (75, 227, 255), "01"),
            Object2D(425, 180, 112, 112, (255, 120, 221), "02"),
            Object2D(700, 310, 138, 90, (125, 239, 150), "03"),
        ]
        self.original_objects = copy.deepcopy(self.objects)

        # Interaction state
        self.grabbed_idx: Optional[int] = None
        self.grab_hand: Optional[str] = None
        self.anchor = (0.0, 0.0)
        self.angular_velocity = 0.0
        self.base_other_pinch: Optional[float] = None
        self.base_size: Optional[Tuple[float, float]] = None
        self.smooth_scale = 1.0
        self.prev_angle: Optional[float] = None
        self.lost_elapsed = 0.0
        self.open_elapsed = 0.0
        self.hover_idx: Optional[int] = None
        self.physics_paused = False

        self.prev_time = time.perf_counter()

    def resize_display(self):
        """Resize the *window*, not the tracking canvas or camera crop."""
        width = max(480, int(CFG.CAM_WIDTH * self.window_scale))
        height = max(300, int(CFG.CAM_HEIGHT * self.window_scale))
        cv2.resizeWindow(self.window_name, width, height)

    def _set_anchor(self, hand: HandData):
        obj = self.objects[self.grabbed_idx]
        cx, cy = obj.center
        dx, dy = hand.pinch_point[0] - cx, hand.pinch_point[1] - cy
        u, v = obj.axes()
        self.anchor = ((dx * u[0] + dy * u[1]) / obj.w,
                       (dx * v[0] + dy * v[1]) / obj.h)
        self.prev_angle = hand.angle

    def _start_grab(self, idx: int, hand: HandData, other: Optional[HandData], hand_label: str):
        self.grabbed_idx = idx
        self.grab_hand = hand_label
        obj = self.objects[idx]
        self._set_anchor(hand)
        self.base_size = (obj.w, obj.h)
        self.smooth_scale = 1.0
        self.base_other_pinch = other.pinch_distance if other else None
        self.open_elapsed = self.lost_elapsed = 0.0
        obj.reset_velocity()
        self.objects.append(self.objects.pop(idx))
        self.grabbed_idx = len(self.objects) - 1

    def _release(self, hand: HandData):
        obj = self.objects[self.grabbed_idx]
        obj.vx = hand.velocity[0] * CFG.THROW_MULTIPLIER
        obj.vy = hand.velocity[1] * CFG.THROW_MULTIPLIER
        obj.va = self.angular_velocity * CFG.SPIN_MULTIPLIER
        self._clear_grab()

    def _update_held_object(self, hand: HandData, other: Optional[HandData], dt: float):
        obj = self.objects[self.grabbed_idx]
        cx, cy = obj.center
        old_u, old_v = obj.axes()
        current_grab = (
            cx + self.anchor[0] * obj.w * old_u[0] + self.anchor[1] * obj.h * old_v[0],
            cy + self.anchor[0] * obj.w * old_u[1] + self.anchor[1] * obj.h * old_v[1],
        )
        if self.prev_angle is not None:
            delta = (hand.angle - self.prev_angle + 180) % 360 - 180
            obj.angle = soft_snap(obj.angle + delta * 0.96, dt)
            self.angular_velocity = delta / max(dt, 0.001)
        self.prev_angle = hand.angle

        if other is not None:
            if self.base_other_pinch is None or self.base_other_pinch < 0.015:
                self.base_other_pinch = max(other.pinch_distance, 0.02)
                self.base_size = (obj.w, obj.h)
                self.smooth_scale = 1.0
            else:
                target = max(CFG.MIN_SCALE, min(CFG.MAX_SCALE,
                             other.pinch_distance / self.base_other_pinch))
                alpha = 1 - 0.8 ** (dt * CFG.REFERENCE_FPS)
                self.smooth_scale += (target - self.smooth_scale) * alpha
                obj.w = self.base_size[0] * self.smooth_scale
                obj.h = self.base_size[1] * self.smooth_scale
        else:
            self.base_other_pinch = None
            self.base_size = (obj.w, obj.h)
            self.smooth_scale = 1.0

        # Transform the original local grab point through scale and rotation.
        u, v = obj.axes()
        ax, ay = self.anchor[0] * obj.w, self.anchor[1] * obj.h
        px, py = hand.pinch_point
        prediction = CFG.PREDICTION / CFG.REFERENCE_FPS
        follow = 1 - (1 - CFG.HOLD_FOLLOW) ** (dt * CFG.REFERENCE_FPS)
        grab_x = current_grab[0] + (px + hand.velocity[0] * prediction - current_grab[0]) * follow
        grab_y = current_grab[1] + (py + hand.velocity[1] * prediction - current_grab[1]) * follow
        obj.x = grab_x - ax * u[0] - ay * v[0] - obj.w * 0.5
        obj.y = grab_y - ax * u[1] - ay * v[1] - obj.h * 0.5
        obj.reset_velocity()
        constrain_to_canvas(obj, CFG.CAM_WIDTH, CFG.CAM_HEIGHT)

    def handle_interaction(self, right: Optional[HandData], left: Optional[HandData],
                           dt: float = 1.0 / 30.0):
        """Lock a grab to one tracked hand, tolerating brief detection gaps."""
        self.hover_idx = None
        hands = {"Right": right, "Left": left}
        if self.grabbed_idx is None:
            for label, hand in hands.items():
                if hand is None:
                    continue
                for i in range(len(self.objects) - 1, -1, -1):
                    if not self.objects[i].contains(*hand.pinch_point, CFG.GRAB_PADDING):
                        continue
                    if self.hover_idx is None:
                        self.hover_idx = i
                    if hand.is_pinching:
                        self._start_grab(i, hand, hands["Left" if label == "Right" else "Right"], label)
                        return
                    break
            return

        owner = hands.get(self.grab_hand)
        other = hands.get("Left" if self.grab_hand == "Right" else "Right")
        if owner is None:
            self.lost_elapsed += dt
            self.open_elapsed = 0.0
            self.angular_velocity = 0.0
            self.base_other_pinch = None
            if self.lost_elapsed + 1e-9 >= CFG.LOST_SECONDS:
                self.objects[self.grabbed_idx].reset_velocity()
                self._clear_grab()
            return
        if self.lost_elapsed > 0 or owner.reacquired:
            # Re-anchor at the new cursor without moving or throwing the object.
            self._set_anchor(owner)
            self.angular_velocity = 0.0
            self.lost_elapsed = 0.0
            self.base_other_pinch = None
            self.objects[self.grabbed_idx].reset_velocity()
            if not owner.is_pinching:
                self._clear_grab()
            return
        self.open_elapsed = 0.0 if owner.is_pinching else self.open_elapsed + dt
        if self.open_elapsed + 1e-9 >= CFG.RELEASE_SECONDS:
            self._release(owner)
        else:
            self._update_held_object(owner, other, dt)

    def _clear_grab(self):
        self.hover_idx = None
        self.grabbed_idx = None
        self.grab_hand = None
        self.base_other_pinch = None
        self.base_size = None
        self.prev_angle = None
        self.open_elapsed = self.lost_elapsed = 0.0
        self.angular_velocity = 0.0
        self.anchor = (0.0, 0.0)
        self.smooth_scale = 1.0

    def run(self):
        try:
            self._run_loop()
        finally:
            self.cleanup()

    def _run_loop(self):
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

            now = time.perf_counter()
            elapsed = max(now - self.prev_time, 0.001)
            self.prev_time = now
            dt = min(elapsed, CFG.MAX_PHYSICS_DT)
            hands = {"Right": right, "Left": left}
            calibration = self.tracker.calibration
            calibration.update(hands, elapsed)
            if calibration.active:
                self._clear_grab()
            else:
                self.handle_interaction(right, left, elapsed)
            update_physics(self.objects, w, h, self.physics_paused or calibration.active,
                           self.grabbed_idx, dt)

            # Render
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
            fps = 1.0 / elapsed

            status = "PAUSED" if self.physics_paused else ("HOLDING" if self.grabbed_idx is not None else "READY")
            if self.lost_elapsed > 0:
                status = "HAND LOST"
            if calibration.active:
                status = "CALIBRATE"
            angle = self.objects[self.grabbed_idx].angle if self.grabbed_idx is not None else 0
            draw_hud(frame, status, fps, self.smooth_scale, angle,
                     self.grabbed_idx is not None, right is not None, left is not None, self.physics_paused)

            cv2.putText(frame, calibration.message, (22, h - 62),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.43, (241, 246, 250), 1, cv2.LINE_AA)
            cv2.imshow(self.window_name, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:                       # ESC
                break
            if key in (ord("c"), ord("C")):
                self._clear_grab()
                calibration.advance(hands)
            if key in (ord("x"), ord("X")):
                self._clear_grab()
                calibration.reset()
                self.tracker.smooth_state.clear()
            if key in (ord("r"), ord("R")):     # Reset
                self.objects = copy.deepcopy(self.original_objects)
                self._clear_grab()
            if key == 32:                       # Space – pause physics
                self.physics_paused = not self.physics_paused
            if key in (ord("-"), ord("_")):
                self.window_scale = max(0.55, round(self.window_scale - 0.1, 2))
                self.resize_display()
            if key in (ord("+"), ord("=")):
                self.window_scale = min(1.4, round(self.window_scale + 0.1, 2))
                self.resize_display()

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
