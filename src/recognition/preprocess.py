import cv2
import mediapipe as mp
import numpy as np
import time
from pathlib import Path

try:
    mp_holistic = mp.solutions.holistic
except Exception:  # pragma: no cover - compatibility fallback
    try:
        from mediapipe.solutions import holistic as mp_holistic
    except Exception:  # pragma: no cover - fully degraded fallback
        mp_holistic = None


class HandKeypointExtractor:
    def __init__(self, static_mode=False, min_detection_confidence=0.5):
        self.mp_holistic = mp_holistic
        self.holistic = None
        self.pose_landmarker = None
        self.hand_landmarker = None
        self._timestamp_ms = 0
        if self.mp_holistic is not None:
            self.holistic = self.mp_holistic.Holistic(
                static_image_mode=static_mode,
                min_detection_confidence=min_detection_confidence,
                model_complexity=1
            )
        else:
            # MediaPipe 1.0 uses the Tasks API and external .task model assets.
            # Use the project-local assets installed for ISL extraction instead
            # of silently returning zero landmarks.
            project_dir = Path(__file__).resolve().parents[2]
            pose_model = project_dir / "models" / "mediapipe" / "pose_landmarker_full.task"
            hand_model = project_dir / "models" / "mediapipe" / "hand_landmarker.task"
            if pose_model.is_file() and hand_model.is_file() and hasattr(mp, "tasks"):
                vision = mp.tasks.vision
                running_mode = vision.RunningMode.VIDEO
                self.pose_landmarker = vision.PoseLandmarker.create_from_options(
                    vision.PoseLandmarkerOptions(
                        base_options=mp.tasks.BaseOptions(model_asset_path=str(pose_model)),
                        running_mode=running_mode,
                        num_poses=1,
                        min_pose_detection_confidence=min_detection_confidence,
                    )
                )
                self.hand_landmarker = vision.HandLandmarker.create_from_options(
                    vision.HandLandmarkerOptions(
                        base_options=mp.tasks.BaseOptions(model_asset_path=str(hand_model)),
                        running_mode=running_mode,
                        num_hands=2,
                        min_hand_detection_confidence=min_detection_confidence,
                    )
                )

    def extract(self, frame, mirror=False):
        if self.holistic is None and (self.pose_landmarker is None or self.hand_landmarker is None):
            return np.zeros((27, 3), dtype=np.float32)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pose_array = None
        left_hand_array = None
        right_hand_array = None
        if self.holistic is not None:
            results = self.holistic.process(rgb)
            if results.pose_landmarks:
                pose_array = np.array([[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark], dtype=np.float32)
            if results.left_hand_landmarks:
                left_hand_array = np.array([[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark], dtype=np.float32)
            if results.right_hand_landmarks:
                right_hand_array = np.array([[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark], dtype=np.float32)
        else:
            self._timestamp_ms = max(self._timestamp_ms + 1, int(time.monotonic() * 1000))
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            pose_result = self.pose_landmarker.detect_for_video(image, self._timestamp_ms)
            hand_result = self.hand_landmarker.detect_for_video(image, self._timestamp_ms)
            if pose_result.pose_landmarks:
                pose_array = np.array([[lm.x, lm.y, lm.z] for lm in pose_result.pose_landmarks[0]], dtype=np.float32)
            for landmarks, handedness in zip(hand_result.hand_landmarks, hand_result.handedness):
                label = handedness[0].category_name.lower() if handedness else ""
                values = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)
                if label == "left":
                    left_hand_array = values
                elif label == "right":
                    right_hand_array = values

        # 1. Parse Pose
        pose_detected = False
        pose_kps = np.zeros((7, 3), dtype=np.float32)

        if pose_array is not None and len(pose_array) >= 33:
            
            # Select 7 nodes: left shoulder (11), right shoulder (12), left elbow (13), right elbow (14), left wrist (15), right wrist (16), nose (0)
            pose_indices = [11, 12, 13, 14, 15, 16, 0]
            for idx, p_idx in enumerate(pose_indices):
                pose_kps[idx] = pose_array[p_idx]
            pose_detected = True

        # Determine normalization parameters
        if pose_detected:
            # Bohacek & Hruz signing-space normalisation
            shoulder_dist = np.sqrt(
                (pose_array[11, 0] - pose_array[12, 0])**2 +
                (pose_array[11, 1] - pose_array[12, 1])**2
            )
            head_metric = shoulder_dist / 2.0
            if head_metric < 1e-5:
                head_metric = 0.15
            center_x, center_y, center_z = pose_array[0]
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
        if left_hand_array is not None and len(left_hand_array) >= 21:
            for idx, h_idx in enumerate(hand_indices):
                left_hand_kps[idx] = left_hand_array[h_idx]
            left_hand_valid = True

        right_hand_kps = np.zeros((10, 3), dtype=np.float32)
        right_hand_valid = False
        if right_hand_array is not None and len(right_hand_array) >= 21:
            for idx, h_idx in enumerate(hand_indices):
                right_hand_kps[idx] = right_hand_array[h_idx]
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
