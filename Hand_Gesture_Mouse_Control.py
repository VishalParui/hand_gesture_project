import cv2
import mediapipe as mp
import numpy as np
import pyautogui

mp_draw = mp.solutions.drawing_utils
mp_hand = mp.solutions.hands

video = cv2.VideoCapture(0)

screen_width, screen_height = pyautogui.size()
screen_width_half = screen_width // 2
screen_height_half = screen_height // 2

# Function to map a value from one range to another
def map_value(value, from_min, from_max, to_min, to_max):
    return (value - from_min) * (to_max - to_min) / (from_max - from_min) + to_min

# Function to check if the thumb and index finger are close
def fingers_close(hand_landmarks):
    thumb_tip = np.array([hand_landmarks.landmark[4].x, hand_landmarks.landmark[4].y])
    index_finger_tip = np.array([hand_landmarks.landmark[8].x, hand_landmarks.landmark[8].y])
    distance = np.linalg.norm(thumb_tip - index_finger_tip)
    if distance < 0.05:  # Adjust this threshold as needed
        return True
    return False

with mp_hand.Hands(min_detection_confidence=0.5,
                   min_tracking_confidence=0.5) as hands:
    while True:
        ret, image = video.read()
        image = cv2.flip(image, 1)  # Flip the camera from inverted
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for id, lm in enumerate(hand_landmarks.landmark):
                    h, w, c = image.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    if id == 8:  # Index finger tip
                        # Convert the finger tip position to screen coordinates
                        mouse_x = int(map_value(cx, 0, w, 0, screen_width))
                        mouse_y = int(map_value(cy, 0, h, 0, screen_height))
                        # Move the mouse cursor
                        pyautogui.moveTo(mouse_x, mouse_y)

                if fingers_close(hand_landmarks):
                    pyautogui.click()

                mp_draw.draw_landmarks(image, hand_landmarks, mp_hand.HAND_CONNECTIONS)

        cv2.imshow("Frame", image)
        k = cv2.waitKey(1)
        if k == ord('q'):
            break

video.release()
cv2.destroyAllWindows()
