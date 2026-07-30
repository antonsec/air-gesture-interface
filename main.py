import cv2
import mediapipe as mp
import numpy as np
import time
import math
import copy

# ==========================
# CORE SETTINGS
# ==========================
CAM_W, CAM_H = 960, 540
PINCH_THRES = 0.050
MIN_SCALE, MAX_SCALE = 0.38, 2.7

# Tracking quality
BASE_SMOOTH = 0.42          # base cursor smoothing
FAST_SMOOTH = 0.78          # when hand is moving fast
SPEED_REF = 28.0            # px/frame where we switch to fast mode
PRED_STRENGTH = 0.45        # how much we predict ahead while grabbing
HOLD_FOLLOW = 0.93          # how tightly object sticks to hand (0.9–0.97)

# Physics (kept light)
FRICTION = 0.96
ANGULAR_FRICTION = 0.93
BOUNCE = 0.68
GRAVITY = 0.32
THROW_MULT = 1.25
SPIN_MULT = 0.85

LOST_TOLERANCE = 8
RELEASE_FRAMES = 3

# ==========================
# CAMERA
# ==========================
cam = cv2.VideoCapture(0)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)
# cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

# ==========================
# MEDIAPIPE
# ==========================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    model_complexity=0,
    min_detection_confidence=0.48,
    min_tracking_confidence=0.48
)

# ==========================
# OBJECTS
# ==========================
def make_obj(x, y, w, h, color, label):
    return {
        "x": float(x), "y": float(y), "w": float(w), "h": float(h),
        "angle": 0.0, "vx": 0.0, "vy": 0.0, "va": 0.0,
        "color": color, "label": label
    }

objects = [
    make_obj(180, 160, 170, 120, (0, 255, 255), "01"),
    make_obj(520, 130, 150, 150, (255, 0, 220), "02"),
    make_obj(350, 310, 190, 110, (0, 255, 170), "03"),
]
original = copy.deepcopy(objects)

# ==========================
# STATE
# ==========================
grabbed = None
offset = (0.0, 0.0)
base_left_pinch = None
base_size = None
smooth_scale = 1.0

# per-hand smooth state
smooth = {}                 # "Right"/"Left" → {"pt": (x,y), "prev": (x,y), "vel": (vx,vy)}
prev_angle = None
lost_counter = 0
open_counter = 0
hover_id = None

gravity_on = False
physics_paused = False

# ==========================
# HELPERS
# ==========================
def pinch_dist(hand):
    t, i = hand.landmark[4], hand.landmark[8]
    return math.hypot(t.x - i.x, t.y - i.y)

def is_pinching(hand):
    return pinch_dist(hand) < PINCH_THRES

def hand_angle(hand):
    w = hand.landmark[0]
    m = hand.landmark[9]
    return math.degrees(math.atan2(m.y - w.y, m.x - w.x))

def inside(px, py, obj):
    return obj["x"] <= px <= obj["x"] + obj["w"] and obj["y"] <= py <= obj["y"] + obj["h"]

def adaptive_smooth(prev, raw, prev_pt):
    """Higher speed → less smoothing (more responsive)."""
    if prev is None:
        return raw, (0.0, 0.0)
    dx = raw[0] - prev_pt[0]
    dy = raw[1] - prev_pt[1]
    speed = math.hypot(dx, dy)
    t = min(1.0, speed / SPEED_REF)
    alpha = BASE_SMOOTH + (FAST_SMOOTH - BASE_SMOOTH) * t
    sx = prev[0] * (1 - alpha) + raw[0] * alpha
    sy = prev[1] * (1 - alpha) + raw[1] * alpha
    return (sx, sy), (dx, dy)

def draw_obj(img, obj, active=False, hover=False):
    x, y, w, h = obj["x"], obj["y"], obj["w"], obj["h"]
    angle = obj["angle"]
    col = obj["color"]
    cx, cy = x + w * 0.5, y + h * 0.5

    rad = math.radians(angle)
    c, s = math.cos(rad), math.sin(rad)
    dx, dy = w * 0.5, h * 0.5
    corners = np.array([
        [cx + (-dx*c - -dy*s), cy + (-dx*s + -dy*c)],
        [cx + ( dx*c - -dy*s), cy + ( dx*s + -dy*c)],
        [cx + ( dx*c -  dy*s), cy + ( dx*s +  dy*c)],
        [cx + (-dx*c -  dy*s), cy + (-dx*s +  dy*c)],
    ], dtype=np.int32)

    # cheap glow
    for t, a in ((8, 0.13), (4, 0.22)):
        over = img.copy()
        cv2.polylines(over, [corners], True, col, t, cv2.LINE_AA)
        cv2.addWeighted(over, a, img, 1 - a, 0, img)

    over = img.copy()
    cv2.fillPoly(over, [corners], col)
    intens = 0.23 if active else (0.15 if hover else 0.09)
    cv2.addWeighted(over, intens, img, 1 - intens, 0, img)
    cv2.polylines(img, [corners], True, col, 3 if active else 2, cv2.LINE_AA)

    cv2.putText(img, obj["label"], (int(cx)-11, int(cy)+5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 1, cv2.LINE_AA)

def physics_step(objs):
    if physics_paused:
        return
    for o in objs:
        if gravity_on:
            o["vy"] += GRAVITY
        o["vx"] *= FRICTION
        o["vy"] *= FRICTION
        o["va"] *= ANGULAR_FRICTION
        o["x"] += o["vx"]
        o["y"] += o["vy"]
        o["angle"] += o["va"]

        # walls
        if o["x"] < 0:
            o["x"] = 0
            o["vx"] *= -BOUNCE
        if o["y"] < 0:
            o["y"] = 0
            o["vy"] *= -BOUNCE
        if o["x"] + o["w"] > CAM_W:
            o["x"] = CAM_W - o["w"]
            o["vx"] *= -BOUNCE
        if o["y"] + o["h"] > CAM_H:
            o["y"] = CAM_H - o["h"]
            o["vy"] *= -BOUNCE

    # light pairwise separation
    n = len(objs)
    for i in range(n):
        for j in range(i+1, n):
            a, b = objs[i], objs[j]
            if (a["x"] < b["x"]+b["w"] and a["x"]+a["w"] > b["x"] and
                a["y"] < b["y"]+b["h"] and a["y"]+a["h"] > b["y"]):
                acx = a["x"] + a["w"]*0.5
                acy = a["y"] + a["h"]*0.5
                bcx = b["x"] + b["w"]*0.5
                bcy = b["y"] + b["h"]*0.5
                dx, dy = acx - bcx, acy - bcy
                dist = math.hypot(dx, dy) or 1.0
                overlap = 0.22 * (min(a["w"], b["w"]) + min(a["h"], b["h"])) - dist
                if overlap > 0:
                    nx, ny = dx/dist, dy/dist
                    push = overlap * 0.5
                    a["x"] += nx * push
                    a["y"] += ny * push
                    b["x"] -= nx * push
                    b["y"] -= ny * push
                    a["vx"] -= (a["vx"]-b["vx"]) * 0.2
                    a["vy"] -= (a["vy"]-b["vy"]) * 0.2
                    b["vx"] += (a["vx"]-b["vx"]) * 0.2
                    b["vy"] += (a["vy"]-b["vy"]) * 0.2

# ==========================
# MAIN LOOP
# ==========================
prev_time = time.perf_counter()

while True:
    if not cam.grab():
        break
    ok, frame = cam.retrieve()
    if not ok:
        break

    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False
    results = hands.process(rgb)
    rgb.flags.writeable = True

    right = left = None

    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_lms, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            label = handedness.classification[0].label
            label = "Right" if label == "Left" else "Left"   # mirror correction

            raw_idx = (hand_lms.landmark[8].x * w, hand_lms.landmark[8].y * h)
            raw_thumb = (hand_lms.landmark[4].x * w, hand_lms.landmark[4].y * h)

            if label not in smooth:
                smooth[label] = {"pt": raw_idx, "prev": raw_idx, "vel": (0.0, 0.0),
                                 "thumb": raw_thumb}
            else:
                st = smooth[label]
                new_pt, vel = adaptive_smooth(st["pt"], raw_idx, st["prev"])
                st["prev"] = st["pt"]
                st["pt"] = new_pt
                st["vel"] = vel
                # light thumb smooth
                st["thumb"] = (
                    st["thumb"][0]*0.55 + raw_thumb[0]*0.45,
                    st["thumb"][1]*0.55 + raw_thumb[1]*0.45
                )

            data = {
                "pt": smooth[label]["pt"],
                "thumb": smooth[label]["thumb"],
                "vel": smooth[label]["vel"],
                "pinch": is_pinching(hand_lms),
                "pinch_d": pinch_dist(hand_lms),
                "angle": hand_angle(hand_lms)
            }

            if label == "Right":
                right = data
            else:
                left = data

            # dots
            col = (0, 255, 200) if data["pinch"] else (200, 80, 255)
            cx, cy = int(data["pt"][0]), int(data["pt"][1])
            tx, ty = int(data["thumb"][0]), int(data["thumb"][1])
            cv2.circle(frame, (cx, cy), 7, col, -1, cv2.LINE_AA)
            cv2.circle(frame, (tx, ty), 5, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(frame, (cx, cy), 11, (255, 255, 255), 1, cv2.LINE_AA)
    else:
        smooth.clear()

    # -------------------- INTERACTION --------------------
    hover_id = None

    if right is not None:
        lost_counter = 0
        px, py = right["pt"]
        vx, vy = right["vel"]

        if grabbed is None:
            for i, obj in enumerate(objects):
                if inside(px, py, obj):
                    hover_id = i
                    break

            if right["pinch"] and hover_id is not None:
                grabbed = hover_id
                offset = (px - objects[grabbed]["x"], py - objects[grabbed]["y"])
                base_size = (objects[grabbed]["w"], objects[grabbed]["h"])
                smooth_scale = 1.0
                base_left_pinch = left["pinch_d"] if left else None
                open_counter = 0
                prev_angle = right["angle"]
                objects[grabbed]["vx"] = objects[grabbed]["vy"] = objects[grabbed]["va"] = 0.0
                # bring to front
                objects.append(objects.pop(grabbed))
                grabbed = len(objects) - 1
        else:
            obj = objects[grabbed]

            if right["pinch"]:
                open_counter = 0
            else:
                open_counter += 1

            if open_counter >= RELEASE_FRAMES:
                # clean throw
                obj["x"] = px - offset[0]
                obj["y"] = py - offset[1]
                obj["vx"] = vx * THROW_MULT
                obj["vy"] = vy * THROW_MULT
                if prev_angle is not None:
                    d = right["angle"] - prev_angle
                    if d > 180: d -= 360
                    if d < -180: d += 360
                    obj["va"] = d * SPIN_MULT
                grabbed = None
                base_left_pinch = base_size = prev_angle = None
                open_counter = 0
            else:
                # ===== HIGH RESPONSIVENESS HOLD =====
                # predict slightly ahead
                pred_x = px + vx * PRED_STRENGTH
                pred_y = py + vy * PRED_STRENGTH
                target_x = pred_x - offset[0]
                target_y = pred_y - offset[1]

                obj["x"] = obj["x"] * (1 - HOLD_FOLLOW) + target_x * HOLD_FOLLOW
                obj["y"] = obj["y"] * (1 - HOLD_FOLLOW) + target_y * HOLD_FOLLOW
                obj["vx"] = obj["vy"] = obj["va"] = 0.0

                # rotation – direct and clean
                if prev_angle is not None:
                    d = right["angle"] - prev_angle
                    if d > 180: d -= 360
                    if d < -180: d += 360
                    obj["angle"] += d * 0.97
                prev_angle = right["angle"]

                # left hand zoom
                if left is not None:
                    if base_left_pinch is None or base_left_pinch < 0.015:
                        base_left_pinch = max(left["pinch_d"], 0.02)
                        base_size = (obj["w"], obj["h"])
                        smooth_scale = 1.0
                    else:
                        target = max(MIN_SCALE, min(MAX_SCALE, left["pinch_d"] / base_left_pinch))
                        smooth_scale = smooth_scale * 0.82 + target * 0.18
                        nw = base_size[0] * smooth_scale
                        nh = base_size[1] * smooth_scale
                        cx = obj["x"] + obj["w"] * 0.5
                        cy = obj["y"] + obj["h"] * 0.5
                        obj["w"], obj["h"] = nw, nh
                        obj["x"] = cx - nw * 0.5
                        obj["y"] = cy - nh * 0.5
                else:
                    base_left_pinch = None
                    base_size = (obj["w"], obj["h"])
                    smooth_scale = 1.0

                # link line
                ocx = int(obj["x"] + obj["w"]*0.5)
                ocy = int(obj["y"] + obj["h"]*0.5)
                cv2.line(frame, (int(px), int(py)), (ocx, ocy), (0, 255, 220), 1, cv2.LINE_AA)
    else:
        if grabbed is not None:
            lost_counter += 1
            if lost_counter > LOST_TOLERANCE:
                objects[grabbed]["vx"] = objects[grabbed].get("last_vx", 0) * 0.5
                objects[grabbed]["vy"] = objects[grabbed].get("last_vy", 0) * 0.5
                grabbed = None
                base_left_pinch = base_size = prev_angle = None
                open_counter = 0

    physics_step(objects)

    # -------------------- DRAW --------------------
    frame = cv2.addWeighted(frame, 0.85, np.zeros_like(frame), 0.15, 0)

    for i, obj in enumerate(objects):
        draw_obj(frame, obj, active=(i == grabbed), hover=(i == hover_id))

    # HUD
    now = time.perf_counter()
    fps = 1.0 / max(now - prev_time, 1e-6)
    prev_time = now

    cv2.line(frame, (0, 42), (w, 42), (0, 180, 255), 1)
    status = "HOLDING" if grabbed is not None else "READY"
    col = (0, 255, 190) if grabbed is not None else (0, 200, 255)
    mode = "PHYS PAUSED" if physics_paused else ("GRAVITY" if gravity_on else "ZERO-G")

    cv2.putText(frame, "RESPONSIVE MANIPULATOR", (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, f"{status}  {int(fps)} fps", (w-190, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, col, 1, cv2.LINE_AA)
    cv2.putText(frame, mode, (12, 68),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 200), 1, cv2.LINE_AA)

    if grabbed is not None:
        cv2.putText(frame, f"ZOOM {int(smooth_scale*100)}%   ROT {int(objects[grabbed]['angle']%360)}°",
                    (12, 94), cv2.FONT_HERSHEY_SIMPLEX, 0.47, (0, 255, 200), 1, cv2.LINE_AA)

    cv2.putText(frame, "RIGHT: grab / move / rotate / throw    LEFT: zoom    G=gravity  SPACE=pause  R=reset  ESC=quit",
                (12, h-12), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (140, 140, 170), 1, cv2.LINE_AA)

    cv2.imshow("RESPONSIVE MANIPULATOR", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break
    if key in (ord("r"), ord("R")):
        objects = copy.deepcopy(original)
        grabbed = None
        base_left_pinch = base_size = prev_angle = None
        smooth_scale = 1.0
    if key in (ord("g"), ord("G")):
        gravity_on = not gravity_on
    if key == 32:
        physics_paused = not physics_paused

cam.release()
hands.close()
cv2.destroyAllWindows()