import cv2
import mediapipe as mp


class HandKeypointExtractor:
    def __init__(self, static_mode=False, max_hands=2, min_detection_confidence=0.5):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=static_mode,
            max_num_hands=max_hands,
            min_detection_confidence=min_detection_confidence
        )

    def extract(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)
        keypoints = []
        if results.multi_hand_landmarks:
            for hand in results.multi_hand_landmarks:
                for lm in hand.landmark:
                    keypoints.extend([lm.x, lm.y, lm.z])
        return keypoints
