import cv2
import mediapipe as mp
import mediapipe.python.solutions.holistic as mp_holistic
import numpy as np


class HandKeypointExtractor:
    def __init__(self, static_mode=False, min_detection_confidence=0.5):
        self.mp_holistic = mp_holistic
        self.holistic = self.mp_holistic.Holistic(
            static_image_mode=static_mode,
            min_detection_confidence=min_detection_confidence,
            model_complexity=1
        )

    def extract(self, frame, mirror=False):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.holistic.process(rgb)

        # 1. Parse Pose
        pose_detected = False
        nose = None
        left_shoulder = None
        right_shoulder = None
        pose_kps = np.zeros((7, 3), dtype=np.float32)

        if results.pose_landmarks:
            pose_landmarks = results.pose_landmarks.landmark
            nose = pose_landmarks[0]
            left_shoulder = pose_landmarks[11]
            right_shoulder = pose_landmarks[12]
            
            # Select 7 nodes: left shoulder (11), right shoulder (12), left elbow (13), right elbow (14), left wrist (15), right wrist (16), nose (0)
            pose_indices = [11, 12, 13, 14, 15, 16, 0]
            for idx, p_idx in enumerate(pose_indices):
                lm = pose_landmarks[p_idx]
                pose_kps[idx] = [lm.x, lm.y, lm.z]
            pose_detected = True

        # Determine normalization parameters
        if pose_detected and nose is not None and left_shoulder is not None and right_shoulder is not None:
            # Bohacek & Hruz signing-space normalisation
            shoulder_dist = np.sqrt(
                (left_shoulder.x - right_shoulder.x)**2 + 
                (left_shoulder.y - right_shoulder.y)**2
            )
            head_metric = shoulder_dist / 2.0
            if head_metric < 1e-5:
                head_metric = 0.15
            center_x, center_y, center_z = nose.x, nose.y, nose.z
        else:
            head_metric = 0.15
            center_x, center_y, center_z = 0.5, 0.5, 0.0

        # Helper function to normalize
        def normalize_kps(kps, is_valid):
            if not is_valid:
                return np.zeros_like(kps)
            norm_kps = np.zeros_like(kps)
            norm_kps[:, 0] = (kps[:, 0] - center_x) / (6.0 * head_metric)
            norm_kps[:, 1] = (kps[:, 1] - center_y) / (7.0 * head_metric)
            norm_kps[:, 2] = (kps[:, 2] - center_z) / (6.0 * head_metric)
            return norm_kps

        # 2. Parse Hands
        hand_indices = [2, 4, 5, 8, 9, 12, 13, 16, 17, 20]

        left_hand_kps = np.zeros((10, 3), dtype=np.float32)
        left_hand_valid = False
        if results.left_hand_landmarks:
            left_landmarks = results.left_hand_landmarks.landmark
            for idx, h_idx in enumerate(hand_indices):
                lm = left_landmarks[h_idx]
                left_hand_kps[idx] = [lm.x, lm.y, lm.z]
            left_hand_valid = True

        right_hand_kps = np.zeros((10, 3), dtype=np.float32)
        right_hand_valid = False
        if results.right_hand_landmarks:
            right_landmarks = results.right_hand_landmarks.landmark
            for idx, h_idx in enumerate(hand_indices):
                lm = right_landmarks[h_idx]
                right_hand_kps[idx] = [lm.x, lm.y, lm.z]
            right_hand_valid = True

        # Normalize Pose and Hands
        pose_norm = normalize_kps(pose_kps, pose_detected)
        left_hand_norm = normalize_kps(left_hand_kps, left_hand_valid)
        right_hand_norm = normalize_kps(right_hand_kps, right_hand_valid)

        # 3. Horizontal Mirroring
        if mirror:
            # Swap left and right hand normalized data
            left_hand_norm, right_hand_norm = right_hand_norm, left_hand_norm
            # Flip X coordinate for hands
            left_hand_norm[:, 0] = -left_hand_norm[:, 0]
            right_hand_norm[:, 0] = -right_hand_norm[:, 0]
            
            # For pose, swap left and right components and flip X
            # Indices in pose_norm:
            # 0: left shoulder, 1: right shoulder
            # 2: left elbow, 3: right elbow
            # 4: left wrist, 5: right wrist
            # 6: nose
            pose_norm[[0, 1]] = pose_norm[[1, 0]]
            pose_norm[[2, 3]] = pose_norm[[3, 2]]
            pose_norm[[4, 5]] = pose_norm[[5, 4]]
            pose_norm[:, 0] = -pose_norm[:, 0]

        # Concatenate: Left hand (10), Right hand (10), Pose (7) -> (27, 3)
        final_kps = np.concatenate([left_hand_norm, right_hand_norm, pose_norm], axis=0)
        return final_kps
