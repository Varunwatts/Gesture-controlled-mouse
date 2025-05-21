import cv2
import mediapipe as mp
import pyautogui
import numpy as np

# Init
cap = cv2.VideoCapture(0)
hands = mp.solutions.hands.Hands(max_num_hands=1)
draw = mp.solutions.drawing_utils
screen_w, screen_h = pyautogui.size()

gesture_text = ""

def distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)
    
    if result.multi_hand_landmarks:
        for handLms in result.multi_hand_landmarks:
            draw.draw_landmarks(frame, handLms, mp.solutions.hands.HAND_CONNECTIONS)

            # Get landmark coordinates
            lm = handLms.landmark
            index_tip = int(lm[8].x * w), int(lm[8].y * h)
            thumb_tip = int(lm[4].x * w), int(lm[4].y * h)
            middle_tip = int(lm[12].x * w), int(lm[12].y * h)
            ring_tip = int(lm[16].x * w), int(lm[16].y * h)
            pinky_tip = int(lm[20].x * w), int(lm[20].y * h)

            # Move mouse with index
            screen_x = int(lm[8].x * screen_w)
            screen_y = int(lm[8].y * screen_h)
            pyautogui.moveTo(screen_x, screen_y)

            # Left click: Ring + Thumb
            if distance(ring_tip, thumb_tip) < 30:
                pyautogui.click()
                pyautogui.sleep(1)
                gesture_text = "Left Click"

            # Right click: Middle + Thumb
            elif distance(middle_tip, thumb_tip) < 30:
                pyautogui.rightClick()
                pyautogui.sleep(1)
                gesture_text = "Right Click"

            # Exit: Pinky + Thumb
            elif distance(pinky_tip, thumb_tip) < 30:
                gesture_text = "Exit Triggered"
                cv2.putText(frame, "Exiting...", (10, 100), cv2.FONT_HERSHEY_SIMPLEX,
                            1.2, (0, 0, 255), 3)
                cv2.imshow("Gesture Mouse", frame)
                cv2.waitKey(1000)
                cap.release()
                cv2.destroyAllWindows()
                exit()

            else:
                gesture_text = "Move"

            # Draw gesture info
            cv2.putText(frame, f"Gesture: {gesture_text}", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    cv2.imshow("Gesture Mouse", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
