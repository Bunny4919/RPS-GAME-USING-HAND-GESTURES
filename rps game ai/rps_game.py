import cv2
import mediapipe as mp
import random
import time
from collections import deque


class GestureRPS:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("Camera not accessible")

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.6
        )
        self.drawer = mp.solutions.drawing_utils

        self.state = "MENU"
        self.countdown = 3
        self.timer = 0

        self.current_gesture = "NONE"
        self.gesture_history = deque(maxlen=6)

        self.player_locked = None
        self.bot_choice = None

        self.player_score = 0
        self.bot_score = 0
        self.draws = 0
        self.result_counted = False

    # ---------------- HAND GESTURE ----------------
    def detect_gesture(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.hands.process(rgb)

        if not result.multi_hand_landmarks:
            self.current_gesture = "NONE"
            return

        hand = result.multi_hand_landmarks[0]
        lm = hand.landmark

        tips = [4, 8, 12, 16, 20]
        pips = [2, 6, 10, 14, 18]

        extended = 0
        for t, p in zip(tips, pips):
            if lm[t].y < lm[p].y:
                extended += 1

        if extended <= 1:
            gesture = "ROCK"
        elif extended >= 4:
            gesture = "PAPER"
        elif lm[8].y < lm[6].y and lm[12].y < lm[10].y:
            gesture = "SCISSORS"
        else:
            gesture = "NONE"

        self.gesture_history.append(gesture)
        if len(set(self.gesture_history)) == 1:
            self.current_gesture = gesture

        self.drawer.draw_landmarks(frame, hand, self.mp_hands.HAND_CONNECTIONS)

    # ---------------- WINNER ----------------
    def winner(self, p, b):
        if p == b:
            return "DRAW"
        rules = {"ROCK": "SCISSORS", "PAPER": "ROCK", "SCISSORS": "PAPER"}
        return "YOU WIN" if rules[p] == b else "BOT WINS"

    # ---------------- MAIN LOOP ----------------
    def run(self):
        window_name = "Gesture RPS"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            self.detect_gesture(frame)

            h, w, _ = frame.shape

            # -------- MENU --------
            if self.state == "MENU":
                menu_text = "press [space = start]  [q = quit]"
                (mw, mh), _ = cv2.getTextSize(menu_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)

                x = (w - mw) // 2
                y = h - 30

                cv2.putText(
                    frame, menu_text,
                    (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 255), 2
                )

            # -------- COUNTDOWN --------
            elif self.state == "COUNTDOWN":
                cv2.putText(frame, str(self.countdown),
                            (w // 2 - 30, h // 2),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            3, (0, 255, 0), 6)

                if time.time() - self.timer >= 1:
                    self.countdown -= 1
                    self.timer = time.time()

                    if self.countdown == 0 and self.current_gesture != "NONE":
                        self.player_locked = self.current_gesture
                        self.bot_choice = random.choice(
                            ["ROCK", "PAPER", "SCISSORS"]
                        )
                        self.state = "RESULT"
                        self.timer = time.time()
                        self.result_counted = False

            # -------- RESULT --------
            elif self.state == "RESULT":
                result = self.winner(self.player_locked, self.bot_choice)

                if not self.result_counted:
                    if result == "YOU WIN":
                        self.player_score += 1
                    elif result == "BOT WINS":
                        self.bot_score += 1
                    else:
                        self.draws += 1
                    self.result_counted = True

                # User picked (LEFT CENTER, RED, SMALL FONT)
                cv2.putText(
                    frame, f"User Picked: {self.player_locked}",
                    (40, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 0, 255), 2
                )

                cv2.putText(frame, f"BOT: {self.bot_choice}",
                            (40, h // 2 + 40),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (0, 255, 255), 2)

                cv2.putText(frame, result,
                            (w // 2 - 120, 100),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.4, (255, 255, 0), 3)

                if time.time() - self.timer > 3:
                    self.state = "MENU"
                    self.countdown = 3
                    self.player_locked = None

            # -------- HUD --------
            hud = f"Detected: {self.current_gesture}"
            cv2.putText(frame, hud, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 255, 255), 2)

            # -------- SCOREBOARD --------
            score = f"You: {self.player_score}  Bot: {self.bot_score}  Draws: {self.draws}"
            cv2.putText(frame, score,
                        (10, h - 15),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 255, 255), 2)

            cv2.imshow(window_name, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == 32 and self.state == "MENU":
                self.state = "COUNTDOWN"
                self.countdown = 3
                self.timer = time.time()

        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    GestureRPS().run()
