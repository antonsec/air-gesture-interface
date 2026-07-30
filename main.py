import cv2
import mediapipe as mp
import time


# ----------------------------
# Camera
# ----------------------------

cam = cv2.VideoCapture(0)

cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)


# ----------------------------
# MediaPipe Hands
# ----------------------------

mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

mp_draw = mp.solutions.drawing_utils


# ----------------------------
# FPS
# ----------------------------

previous_time = 0


# ----------------------------
# Smooth tracking storage
# ----------------------------

smooth_points = {}


# ----------------------------
# Main loop
# ----------------------------

while True:

    success, frame = cam.read()

    if not success:
        break


    # Mirror camera
    frame = cv2.flip(frame, 1)


    # Convert for MediaPipe
    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )


    results = hands.process(rgb)


    # ----------------------------
    # Hand detection
    # ----------------------------

    if results.multi_hand_landmarks:


        for hand_id, hand_landmarks in enumerate(results.multi_hand_landmarks):


            # Draw skeleton
            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )


            # Hand name
            label = "Hand"


            if results.multi_handedness:

                label = results.multi_handedness[
                    hand_id
                ].classification[0].label



            # Index fingertip landmark
            index = hand_landmarks.landmark[8]


            h, w, _ = frame.shape


            raw_x = int(index.x * w)
            raw_y = int(index.y * h)



            # ----------------------------
            # Smooth movement
            # ----------------------------

            if hand_id not in smooth_points:

                smooth_points[hand_id] = (
                    raw_x,
                    raw_y
                )


            old_x, old_y = smooth_points[hand_id]


            smooth_x = int(
                old_x * 0.8 +
                raw_x * 0.2
            )


            smooth_y = int(
                old_y * 0.8 +
                raw_y * 0.2
            )


            smooth_points[hand_id] = (
                smooth_x,
                smooth_y
            )



            # ----------------------------
            # Futuristic fingertip
            # ----------------------------

            cv2.circle(
                frame,
                (smooth_x, smooth_y),
                15,
                (255,255,0),
                -1
            )


            cv2.circle(
                frame,
                (smooth_x, smooth_y),
                25,
                (255,255,255),
                2
            )


            cv2.putText(
                frame,
                f"{label}  {smooth_x},{smooth_y}",
                (smooth_x - 80, smooth_y - 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255,255,0),
                2
            )



    # ----------------------------
    # FPS
    # ----------------------------

    current_time = time.time()

    fps = 1 / (
        current_time - previous_time
    )

    previous_time = current_time


    cv2.putText(
        frame,
        f"FPS: {int(fps)}",
        (20,40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0,255,0),
        2
    )


    # Display

    cv2.imshow(
        "JARVIS HAND TRACKING",
        frame
    )


    if cv2.waitKey(1) == 27:
        break



cam.release()
cv2.destroyAllWindows()