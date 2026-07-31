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
    CAM_WIDTH: int = 800
    CAM_HEIGHT: int = 450

    # Tracking
    PINCH_THRESHOLD: float = 0.050
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
    MIN_SCALE: float = 0.38
    MAX_SCALE: float = 2.7

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

    def contains(self, px: float, py: float) -> bool:
        return self.x <= px <= self.x + self.w and self.y <= py <= self.y + self.h

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
    thumb = hand_landmarks.landmark[4]
    index = hand_landmarks.landmark[8]
    return math.hypot(thumb.x - index.x, thumb.y - index.y)


def get_hand_angle(hand_landmarks) -> float:
    wrist = hand_landmarks.landmark[0]
    middle = hand_landmarks.landmark[9]
    return math.degrees(math.atan2(middle.y - wrist.y, middle.x - wrist.x))


# ═══════════════════════════════════════════════════════════════
# PHYSICS
# ═══════════════════════════════════════════════════════════════

def update_physics(objects: List[Object2D], frame_w: int, frame_h: int, paused: bool):
    if paused:
        return

    for obj in objects:
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


def draw_hud(img: np.ndarray, status: str, fps: float, scale: float, angle: float, holding: bool):
    h, w = img.shape[:2]
    color = (0, 255, 190) if holding else (0, 200, 255)

    cv2.line(img, (0, 38), (w, 38), (0, 180, 255), 1)
    cv2.putText(img, "OBJECT MANIPULATOR", (10, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(img, f"{status}  {int(fps)} fps", (w - 175, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA)

    if holding:
        cv2.putText(img, f"ZOOM {int(scale * 100)}%  ROT {int(angle % 360)}°",
                    (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 200), 1, cv2.LINE_AA)

    cv2.putText(img, "RIGHT: grab/move/rotate/throw   LEFT: zoom   SPACE=pause   R=reset   ESC=quit",
                (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (140, 140, 170), 1, cv2.LINE_AA)


# ═══════════════════════════════════════════════════════════════
# HAND TRACKER
# ═══════════════════════════════════════════════════════════════

class HandTracker:
    def __init__(self):
        self.hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            model_complexity=0,
            min_detection_confidence=0.45,
            min_tracking_confidence=0.45
        )
        self.smooth_state: Dict[str, dict] = {}

    def process(self, frame: np.ndarray) -> Tuple[Optional[HandData], Optional[HandData]]:
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self.hands.process(rgb)
        rgb.flags.writeable = True

        right = left = None

        if not results.multi_hand_landmarks or not results.multi_handedness:
            self.smooth_state.clear()
            return None, None

        for landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            # Correct for mirrored image
            label = handedness.classification[0].label
            label = "Right" if label == "Left" else "Left"

            raw_index = (landmarks.landmark[8].x * w, landmarks.landmark[8].y * h)
            raw_thumb = (landmarks.landmark[4].x * w, landmarks.landmark[4].y * h)

            if label not in self.smooth_state:
                self.smooth_state[label] = {
                    "pt": raw_index,
                    "prev": raw_index,
                    "vel": (0.0, 0.0),
                    "thumb": raw_thumb
                }
            else:
                state = self.smooth_state[label]
                new_pt, vel = adaptive_smooth(state["pt"], raw_index, state["prev"])
                state["prev"] = state["pt"]
                state["pt"] = new_pt
                state["vel"] = vel
                state["thumb"] = (
                    state["thumb"][0] * 0.55 + raw_thumb[0] * 0.45,
                    state["thumb"][1] * 0.55 + raw_thumb[1] * 0.45
                )

            state = self.smooth_state[label]
            data = HandData(
                index=state["pt"],
                thumb=state["thumb"],
                velocity=state["vel"],
                is_pinching=get_pinch_distance(landmarks) < CFG.PINCH_THRESHOLD,
                pinch_distance=get_pinch_distance(landmarks),
                angle=get_hand_angle(landmarks)
            )

            if label == "Right":
                right = data
            else:
                left = data

        return right, left

    def draw_cursors(self, frame: np.ndarray, right: Optional[HandData], left: Optional[HandData]):
        for hand in (right, left):
            if hand is None:
                continue
            color = (0, 255, 200) if hand.is_pinching else (200, 80, 255)
            cx, cy = int(hand.index[0]), int(hand.index[1])
            tx, ty = int(hand.thumb[0]), int(hand.thumb[1])
            cv2.circle(frame, (cx, cy), 6, color, -1, cv2.LINE_AA)
            cv2.circle(frame, (tx, ty), 4, (255, 255, 255), -1, cv2.LINE_AA)

    def close(self):
        self.hands.close()


# ═══════════════════════════════════════════════════════════════
# MAIN APPLICATION
# ═══════════════════════════════════════════════════════════════

class ObjectManipulator:
    def __init__(self):
        self.cam = cv2.VideoCapture(0)
        self.cam.set(cv2.CAP_PROP_FRAME_WIDTH, CFG.CAM_WIDTH)
        self.cam.set(cv2.CAP_PROP_FRAME_HEIGHT, CFG.CAM_HEIGHT)
        self.cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.tracker = HandTracker()

        self.objects: List[Object2D] = [
            Object2D(150, 140, 160, 110, (0, 255, 255), "01"),
            Object2D(420, 120, 140, 140, (255, 0, 220), "02"),
            Object2D(280, 260, 170, 100, (0, 255, 170), "03"),
        ]
        self.original_objects = copy.deepcopy(self.objects)

        # Interaction state
        self.grabbed_idx: Optional[int] = None
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

    def _start_grab(self, idx: int, hand: HandData, left: Optional[HandData]):
        obj = self.objects[idx]
        self.grabbed_idx = idx
        self.offset = (hand.index[0] - obj.x, hand.index[1] - obj.y)
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
        px, py = hand.index
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
        self.base_left_pinch = None
        self.base_size = None
        self.prev_angle = None
        self.open_counter = 0

    def _update_held_object(self, hand: HandData, left: Optional[HandData]):
        obj = self.objects[self.grabbed_idx]
        px, py = hand.index
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
        self.hover_idx = None

        if right is None:
            if self.grabbed_idx is not None:
                self.lost_counter += 1
                if self.lost_counter > CFG.LOST_TOLERANCE:
                    self.objects[self.grabbed_idx].angle = soft_snap(
                        self.objects[self.grabbed_idx].angle
                    )
                    self.grabbed_idx = None
                    self.base_left_pinch = None
                    self.base_size = None
                    self.prev_angle = None
                    self.open_counter = 0
            return

        self.lost_counter = 0
        px, py = right.index

        if self.grabbed_idx is None:
            # Hover + grab
            for i, obj in enumerate(self.objects):
                if obj.contains(px, py):
                    self.hover_idx = i
                    break

            if right.is_pinching and self.hover_idx is not None:
                self._start_grab(self.hover_idx, right, left)
        else:
            # Currently holding
            if right.is_pinching:
                self.open_counter = 0
            else:
                self.open_counter += 1

            if self.open_counter >= CFG.RELEASE_FRAMES:
                self._release(right)
            else:
                self._update_held_object(right, left)

    def run(self):
        while True:
            if not self.cam.grab():
                break
            ok, frame = self.cam.retrieve()
            if not ok:
                break

            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

            # Tracking
            right, left = self.tracker.process(frame)
            self.tracker.draw_cursors(frame, right, left)

            # Interaction
            self.handle_interaction(right, left)

            # Physics
            update_physics(self.objects, w, h, self.physics_paused)

            # Render
            frame = cv2.addWeighted(frame, 0.87, np.zeros_like(frame), 0.13, 0)

            for i, obj in enumerate(self.objects):
                draw_object(
                    frame, obj,
                    active=(i == self.grabbed_idx),
                    hover=(i == self.hover_idx)
                )

            # Connection line while holding
            if self.grabbed_idx is not None and right is not None:
                obj = self.objects[self.grabbed_idx]
                cx, cy = obj.center
                cv2.line(frame,
                         (int(right.index[0]), int(right.index[1])),
                         (int(cx), int(cy)),
                         (0, 255, 220), 1, cv2.LINE_AA)

            # HUD
            now = time.perf_counter()
            fps = 1.0 / max(now - self.prev_time, 1e-6)
            self.prev_time = now

            status = "HOLDING" if self.grabbed_idx is not None else "READY"
            angle = self.objects[self.grabbed_idx].angle if self.grabbed_idx is not None else 0
            draw_hud(frame, status, fps, self.smooth_scale, angle, self.grabbed_idx is not None)

            cv2.imshow("OBJECT MANIPULATOR", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:                       # ESC
                break
            if key in (ord("r"), ord("R")):     # Reset
                self.objects = copy.deepcopy(self.original_objects)
                self.grabbed_idx = None
                self.base_left_pinch = None
                self.base_size = None
                self.prev_angle = None
                self.smooth_scale = 1.0
            if key == 32:                       # Space – pause physics
                self.physics_paused = not self.physics_paused

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