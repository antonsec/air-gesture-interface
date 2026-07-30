import cv2
import mediapipe as mp
import time

# Camera
cam = cv2.VideoCapture(0)

# MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

mp_drawing = mp.solutions.drawing_utils


# FPS variables
previous_time = 0


while True:

    success, frame = cam.read()

    if not success:
        break

    # Mirror camera
    frame = cv2.flip(frame, 1)


    # Convert BGR -> RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


    # Process hand
    results = hands.process(rgb_frame)


    # Draw hand landmarks
    if results.multi_hand_landmarks:

        print("Hand detected")

        for hand_landmarks in results.multi_hand_landmarks:

            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )


            # Get index finger tip
            index_tip = hand_landmarks.landmark[8]

            height, width, _ = frame.shape

            x = int(index_tip.x * width)
            y = int(index_tip.y * height)


            # Draw fingertip circle
            cv2.circle(
                frame,
                (x, y),
                10,
                (0, 255, 255),
                -1
            )

            cv2.putText(
                frame,
                f"Finger: {x}, {y}",
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0,255,255),
                2
            )


    # FPS counter
    current_time = time.time()

    fps = 1 / (current_time - previous_time)
    previous_time = current_time


    cv2.putText(
        frame,
        f"FPS: {int(fps)}",
        (20, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0,255,0),
        2
    )


    # Display
    cv2.imshow(
        "Tony Stark Prototype",
        frame
    )


    # ESC closes
    key = cv2.waitKey(1)

    if key == 27:
        break


# Cleanup
cam.release()
cv2.destroyAllWindows()