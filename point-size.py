import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os, urllib.request

# MODEL DOWNLOAD
model_path = "hand_landmarker.task"
if not os.path.exists(model_path):
    print("Downloading model...")
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
        model_path
    )

base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
detector = vision.HandLandmarker.create_from_options(options)

def is_pointing(hand):
    index_up  = hand[8].y  < hand[6].y
    middle_down = hand[12].y > hand[10].y
    ring_down   = hand[16].y > hand[14].y
    pinky_down  = hand[20].y > hand[18].y
    return index_up and middle_down and ring_down and pinky_down

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
print("Point both index fingers to draw a line - press Q to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = detector.detect(mp_image)

    fingertips = []

    if result.hand_landmarks:
        for hand in result.hand_landmarks:
            if is_pointing(hand):
                tip = hand[8]  # index fingertip
                x, y = int(tip.x * w), int(tip.y * h)
                fingertips.append((x, y))
                cv2.circle(frame, (x, y), 8, (0, 0, 255), -1)

    if len(fingertips) == 2:
        cv2.line(frame, fingertips[0], fingertips[1], (0, 0, 255), 3)

        mid_x = (fingertips[0][0] + fingertips[1][0]) // 2
        mid_y = (fingertips[0][1] + fingertips[1][1]) // 2
        dist = int(((fingertips[0][0] - fingertips[1][0])**2 +
                    (fingertips[0][1] - fingertips[1][1])**2) ** 0.5)
        cv2.putText(frame, f"{dist}px", (mid_x, mid_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    else:
        cv2.putText(frame, "point both index fingers", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)

    cv2.imshow("", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()