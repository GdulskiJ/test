import cv2
import mediapipe as mp
import pyautogui
from collections import deque
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MODEL_PATH = r"models/hand_landmarker.task"

annotated_image = None
hand_landmarks_list = []

# Callback - zapisuje wykryte dłonie i obraz
def callback(result, output_image, timestamp_ms):
    global annotated_image, hand_landmarks_list
    annotated_image = output_image
    hand_landmarks_list = []

    if result.hand_landmarks:
        for hand in result.hand_landmarks:
            hand_landmarks_list.append(hand)  # hand jest listą landmarków

# Konfiguracja detektora
BaseOptions = python.BaseOptions
HandLandmarker = vision.HandLandmarker
HandLandmarkerOptions = vision.HandLandmarkerOptions
VisionRunningMode = vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=callback,
    num_hands=1
)

landmarker = HandLandmarker.create_from_options(options)

# Połączenia dłoni
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20)
]

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
screen_w, screen_h = pyautogui.size()
timestamp = 0

# Kolejka do wygładzania kursora
finger_history = deque(maxlen=6)

# Wykrywa które palce są uniesione
def fingers_up(hand):
    """
    Zwraca listę palców [kciuk, wskazujący, środkowy, serdeczny, mały]
    True = palec uniesiony, False = palec zgięty
    """
    tips_ids = [4,8,12,16,20]
    # Lustrzane odbicie X
    points = [(1-lm.x, lm.y) for lm in hand]  # odwracamy X
    fingers = []

    # Kciuk
    fingers.append(points[4][0] > points[3][0])

    # Pozostałe palce
    for tip_id in tips_ids[1:]:
        fingers.append(points[tip_id][1] < points[tip_id - 2][1])

    return fingers, points

# Sprawdza czy palec się porusza powyżej threshold
def is_finger_moving(history, threshold=0.01):
    if len(history) < 2:
        return True
    dx = abs(history[-1][0] - history[-2][0])
    dy = abs(history[-1][1] - history[-2][1])
    return dx > threshold or dy > threshold

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    # lustrzane odbicie obrazu
    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (256,256))
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    landmarker.detect_async(mp_image, timestamp)
    timestamp += 1

    display_frame = frame.copy()

    if annotated_image is not None:
        display_frame = cv2.cvtColor(annotated_image.numpy_view(), cv2.COLOR_RGB2BGR)

        for hand in hand_landmarks_list:
            fingers, points = fingers_up(hand)

            # rysowanie punktów i linii
            for px, py in points:
                cv2.circle(display_frame, (int(px*256), int(py*256)), 3, (0,255,0), -1)
            for start, end in HAND_CONNECTIONS:
                cv2.line(display_frame,
                         (int(points[start][0]*256), int(points[start][1]*256)),
                         (int(points[end][0]*256), int(points[end][1]*256)),
                         (255,0,0), 2)

            finger_tip = points[8]  # palec wskazujący
            finger_history.append(finger_tip)
            action = None

            # Ruch myszy tylko jeśli kciuk opuszczony
            # Ruch myszy tylko jeśli kciuk opuszczony
            if fingers[1] and not fingers[0]:
                avg_x = sum(p[0] for p in finger_history) / len(finger_history)
                avg_y = sum(p[1] for p in finger_history) / len(finger_history)

                # Lustrzane odbicie X przy mapowaniu na ekran
                screen_x = int((1 - avg_x) * screen_w)  # <- to powoduje, że ruch palca w lewo idzie w lewo
                screen_y = int(avg_y * screen_h)

                pyautogui.moveTo(screen_x, screen_y)
                action = "Ruch myszy"

            # Kliknięcie OK (kciuk + wskazujący, palec stabilny)
            elif fingers[0] and fingers[1] and not is_finger_moving(finger_history, threshold=0.01):
                pyautogui.click()
                action = "Kliknięcie (OK)"
                # Pięść (wszystkie palce zgięte)
            elif sum(fingers) == 0:
                action = "Pięść wykryta (pauza ruchu myszy)"
                # tutaj możesz np. zatrzymać ruch myszy lub wykonać inną akcję

            if action:
                print(f"Akcja: {action}")

    cv2.imshow("Hand Mouse Control", display_frame)
    cv2.waitKey(1)