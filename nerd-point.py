import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os, urllib.request
from PIL import Image, ImageDraw, ImageFont
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
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
detector = vision.HandLandmarker.create_from_options(options)

try:
    emoji_font = ImageFont.truetype("seguiemj.ttf", 80)
except:
    emoji_font = ImageFont.load_default()

def is_pointing_up(hand):
    index_up    = hand[8].y  < hand[6].y
    middle_down = hand[12].y > hand[10].y
    ring_down   = hand[16].y > hand[14].y
    pinky_down  = hand[20].y > hand[18].y
    return index_up and middle_down and ring_down and pinky_down

def draw_emoji(frame, emoji, x, y):
    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    draw.text((x, y), emoji, font=emoji_font, embedded_color=True)
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("Point up with one finger - press Q to quit")

cv2.namedWindow("nerd-point", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("nerd-point", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

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
        for hand in result.hand_landmarks:
            if is_pointing_up(hand):
                tip = hand[8]
                x = int(tip.x * w) - 40
                y = int(tip.y * h) - 100 
                frame = draw_emoji(frame, "🤓", x, y)
            else:
                cv2.putText(frame, "point up!", (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)

    cv2.imshow("nerd-point", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()