import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os, urllib.request
import numpy as np

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
    index_up    = hand[8].y  < hand[6].y
    middle_down = hand[12].y > hand[10].y
    ring_down   = hand[16].y > hand[14].y
    pinky_down  = hand[20].y > hand[18].y
    return index_up and middle_down and ring_down and pinky_down

def dist(p1, p2):
    return int(((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2) ** 0.5)

def dist_to_colour(d, max_dist):
    ratio = min(d / max_dist, 1.0)
    hue = int(ratio * 179)
    hsv_colour = np.uint8([[[hue, 255, 255]]])
    bgr = cv2.cvtColor(hsv_colour, cv2.COLOR_HSV2BGR)
    return tuple(int(x) for x in bgr[0][0])

cap = cv2.VideoCapture(0)
print("Point both index fingers and move them apart to change the mood - press Q to quit")

MAX_DIST = 600
current_colour = (255, 255, 255)

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
                tip = hand[8]
                fingertips.append((int(tip.x * w), int(tip.y * h)))

    if len(fingertips) == 2:
        d = dist(fingertips[0], fingertips[1])
        current_colour = dist_to_colour(d, MAX_DIST)

        cv2.circle(frame, fingertips[0], 8, current_colour, -1)
        cv2.circle(frame, fingertips[1], 8, current_colour, -1)
        cv2.line(frame, fingertips[0], fingertips[1], current_colour, 3)
    else:
        cv2.putText(frame, "point both index fingers", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)

    overlay = np.full_like(frame, current_colour, dtype=np.uint8)
    frame = cv2.addWeighted(frame, 0.75, overlay, 0.25, 0)

    cv2.imshow("point-colour", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()