import cv2
import mediapipe as mp


class HandTrackingDemo:
    def __init__(self, static_mode=False, max_hands=2,
                 min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_styles = mp.solutions.drawing_styles
        self.hands = self.mp_hands.Hands(
            static_image_mode=static_mode,
            max_num_hands=max_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )

    def process_frame(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_styles.get_default_hand_landmarks_style(),
                    self.mp_styles.get_default_hand_connections_style()
                )
        return frame, results

    def run_webcam(self, camera_id=0):
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open camera {camera_id}")
        print("Press 'q' to quit, 's' to save frame")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            annotated, results = self.process_frame(frame)
            n_hands = len(results.multi_hand_landmarks) if results.multi_hand_landmarks else 0
            cv2.putText(annotated, f"Hands: {n_hands}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("MediaPipe Hand Tracking - AITE", annotated)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                cv2.imwrite("hand_frame.jpg", annotated)
                print("Frame saved")
        cap.release()
        cv2.destroyAllWindows()

    def process_video(self, video_path, output_path=None):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video {video_path}")
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            annotated, _ = self.process_frame(frame)
            if out:
                out.write(annotated)
            frame_count += 1
        cap.release()
        if out:
            out.release()
        print(f"Processed {frame_count} frames")
        return frame_count


if __name__ == "__main__":
    import sys
    demo = HandTrackingDemo()
    if len(sys.argv) > 1:
        demo.process_video(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
    else:
        demo.run_webcam()
