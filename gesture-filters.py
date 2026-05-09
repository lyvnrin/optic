import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
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
    num_hands=1,
    min_hand_detection_confidence=0.7,
    min_tracking_confidence=0.6
)
detector = vision.HandLandmarker.create_from_options(options)

# FILTERS
def filter_original(frame):
    return frame.copy()

def filter_bw(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

def filter_warm(frame):
    warm = frame.copy().astype(np.float32)
    warm[:, :, 2] = np.clip(warm[:, :, 2] * 1.3, 0, 255)
    warm[:, :, 0] = np.clip(warm[:, :, 0] * 0.7, 0, 255)
    return warm.astype(np.uint8)

def filter_cool(frame):
    cool = frame.copy().astype(np.float32)
    cool[:, :, 0] = np.clip(cool[:, :, 0] * 1.3, 0, 255)
    cool[:, :, 2] = np.clip(cool[:, :, 2] * 0.7, 0, 255)
    return cool.astype(np.uint8)

def filter_sepia(frame):
    f = frame.astype(np.float32)
    r = np.clip(f[:,:,2]*0.393 + f[:,:,1]*0.769 + f[:,:,0]*0.189, 0, 255)
    g = np.clip(f[:,:,2]*0.349 + f[:,:,1]*0.686 + f[:,:,0]*0.168, 0, 255)
    b = np.clip(f[:,:,2]*0.272 + f[:,:,1]*0.534 + f[:,:,0]*0.131, 0, 255)
    return np.stack([b, g, r], axis=2).astype(np.uint8)

def filter_vignette(frame):
    h, w = frame.shape[:2]
    X = cv2.getGaussianKernel(w, w * 0.5)
    Y = cv2.getGaussianKernel(h, h * 0.5)
    mask = Y @ X.T
    mask = mask / mask.max()
    return (frame * mask[:, :, np.newaxis]).astype(np.uint8)

FILTERS = {
    "Original": filter_original,
    "B&W":      filter_bw,
    "Warm":     filter_warm,
    "Cool":     filter_cool,
    "Sepia":    filter_sepia,
    "Vignette": filter_vignette,
}

# GESTURE DETECTION
def count_fingers(hand):
    """Count extended fingers using landmark y positions."""
    TIPS = [8, 12, 16, 20]
    PIPS = [6, 10, 14, 18]

    fingers_up = sum(hand[tip].y < hand[pip].y for tip, pip in zip(TIPS, PIPS))

    # Thumb: use x-axis
    if hand[4].x < hand[3].x:
        fingers_up += 1

    return fingers_up

def gesture_to_filter(n):
    return ["Original", "B&W", "Warm", "Cool", "Sepia", "Vignette"][min(n, 5)]

# OVERLAY
def draw_overlay(frame, filter_name, n_fingers):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - 60), (w, h), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)

    label = f"Filter: {filter_name}  ({n_fingers} finger{'s' if n_fingers != 1 else ''})"
    cv2.putText(frame, label, (12, h - 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)

    hints = ["0:Original", "1:B&W", "2:Warm", "3:Cool", "4:Sepia", "5:Vignette"]
    for i, hint in enumerate(hints):
        cv2.putText(frame, hint, (w - 160, 24 + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
    return frame

cap = cv2.VideoCapture(0)

current_filter = "Original"
n_fingers = 0

print("Running - press Q to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = detector.detect(mp_image)

    if result.hand_landmarks:
        hand = result.hand_landmarks[0]
        n_fingers = count_fingers(hand)
        current_filter = gesture_to_filter(n_fingers)

        # DRAWING THE LANDMARKS
        for lm in hand:
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame, (cx, cy), 3, (0, 255, 180), -1)

    filtered = FILTERS[current_filter](frame)
    display  = draw_overlay(filtered, current_filter, n_fingers)

    cv2.imshow("gesture-filters", display)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()