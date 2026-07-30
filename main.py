import cv2
import mediapipe as mp
import time


# Camera
cam = cv2.VideoCapture(0)

cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)


# MediaPipe

mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

mp_draw = mp.solutions.drawing_utils


# Drawing canvas
canvas = None


# Previous finger position
previous_points = {}


# FPS
previous_time = 0



while True:


    success, frame = cam.read()

    if not success:
        break


    frame = cv2.flip(frame,1)


    # Create drawing layer once
    if canvas is None:
        canvas = frame.copy()
        canvas[:] = 0



    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )


    results = hands.process(rgb)



    if results.multi_hand_landmarks:


        for hand_id, hand_landmarks in enumerate(results.multi_hand_landmarks):


            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )


            # Index finger tip
            finger = hand_landmarks.landmark[8]


            h,w,_ = frame.shape


            x = int(finger.x*w)
            y = int(finger.y*h)



            # Smooth point

            if hand_id in previous_points:

                old_x, old_y = previous_points[hand_id]


                cv2.line(
                    canvas,
                    (old_x,old_y),
                    (x,y),
                    (255,255,0),
                    5
                )


            previous_points[hand_id] = (x,y)



            # Finger glow

            cv2.circle(
                frame,
                (x,y),
                15,
                (255,255,0),
                -1
            )



    # Combine drawing + camera

    frame = cv2.addWeighted(
        frame,
        0.7,
        canvas,
        0.8,
        0
    )



    # FPS

    current_time=time.time()

    fps=1/(current_time-previous_time)

    previous_time=current_time


    cv2.putText(
        frame,
        f"FPS {int(fps)}",
        (20,40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0,255,0),
        2
    )


    cv2.imshow(
        "JARVIS AIR DRAW",
        frame
    )


    key=cv2.waitKey(1)


    # ESC quit
    if key==27:
        break


    # C clears drawing
    if key==ord("c"):
        canvas[:] = 0



cam.release()
cv2.destroyAllWindows()