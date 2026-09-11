import argparse
import cv2
import json
import sys
import numpy as np
from pathlib import Path

# ---------------- Display ---------------- #

WINDOW_NAME = "ISL Landmark Animation"

CANVAS_W = 900
CANVAS_H = 900

DISPLAY_W = 900

VIEW_MARGIN_X = 80
VIEW_MARGIN_TOP = 120
VIEW_MARGIN_BOTTOM = 50
MAX_VIEW_SCALE = 350

# ---------------- Connections ---------------- #

POSE_CONNECTIONS = [
    (11,13), (13,15),
    (12,14), (14,16),
    (11,12),
    (11,23), (12,24),
    (23,24),
    (23,25), (25,27),
    (24,26), (26,28),
]

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),
    (0,17)
]

# ------------------------------------------------ #

def project(pt, view):
    x, y, z = pt
    scale, center_x, center_y = view
    u = int(CANVAS_W / 2 + (x - center_x) * scale)
    v = int((VIEW_MARGIN_TOP + CANVAS_H - VIEW_MARGIN_BOTTOM) / 2 + (y - center_y) * scale)
    return (u, v)


def draw_landmarks(canvas, landmarks, color, view):
    for p in landmarks:
        cv2.circle(canvas, project(p, view), 4, color, -1)


def draw_connections(canvas, landmarks, connections, color, view):
    for a, b in connections:
        cv2.line(
            canvas,
            project(landmarks[a], view),
            project(landmarks[b], view),
            color,
            2
        )


def fitted_view(frames):
    """Fit the complete sequence into one stable viewport.

    Landmark clips were extracted with different coordinate ranges. Fitting
    once for the whole clip avoids a zoomed/cropped signer while keeping the
    camera stable during the sign.
    """
    points = [
        point
        for frame in frames
        if not frame.get("is_word_gap")
        for group in ("pose", "left_hand", "right_hand")
        for point in frame.get(group, [])
        if len(point) >= 2
    ]
    if not points:
        return (MAX_VIEW_SCALE, 0.0, 0.0)
    xs = [float(point[0]) for point in points]
    ys = [float(point[1]) for point in points]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    span_x = max(max_x - min_x, 1e-6)
    span_y = max(max_y - min_y, 1e-6)
    available_width = CANVAS_W - 2 * VIEW_MARGIN_X
    available_height = CANVAS_H - VIEW_MARGIN_TOP - VIEW_MARGIN_BOTTOM
    scale = min(available_width / span_x, available_height / span_y, MAX_VIEW_SCALE)
    return (scale, (min_x + max_x) / 2, (min_y + max_y) / 2)


def fitted_views_by_token(frames):
    """Return stable fitted views for each token in a composite sequence."""
    grouped = {}
    for frame in frames:
        token = frame.get("token_label")
        if token and not frame.get("is_word_gap"):
            grouped.setdefault(token, []).append(frame)
    return {token: fitted_view(token_frames) for token, token_frames in grouped.items()}


def process_file(json_path_or_data, save_video=None, no_show=False, override_text=None, playback_fps=None):
    # Accept either a file path or an already-loaded dict (e.g. from fingerspell)
    if isinstance(json_path_or_data, dict):
        data = json_path_or_data
        gloss = override_text if override_text else "fingerspell"
    else:
        json_path = json_path_or_data
        with open(json_path, "r") as f:
            data = json.load(f)
        gloss = override_text if override_text else json_path.stem

    frames = data["frames"]
    default_view = fitted_view(frames)
    token_views = fitted_views_by_token(frames)
    source_fps = data.get("fps", 30)
    fps = float(playback_fps) if playback_fps else source_fps
    if fps <= 0:
        raise ValueError("playback fps must be positive")
    delay = max(1, int(1000 / fps))

    print(f"\nPlaying: {gloss}")
    print(f"Frames : {len(frames)}")

    writer = None
    if save_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(save_video, fourcc, fps, (CANVAS_W, CANVAS_H))
        print(f"Saving video to: {save_video}")

    if not no_show:
        try:
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(WINDOW_NAME, DISPLAY_W, DISPLAY_W)
        except Exception as e:
            print(f"GUI window unavailable: {e}. Running in headless mode.")
            no_show = True

    frame_idx = 0

    while frame_idx < len(frames):
        canvas = np.zeros((CANVAS_H, CANVAS_W, 3), dtype=np.uint8)
        frame = frames[frame_idx]
        view = token_views.get(frame.get("token_label"), default_view)

        # ---------------- Text ---------------- #
        cv2.putText(
            canvas,
            gloss,
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (255,255,255),
            2,
            cv2.LINE_AA
        )

        cv2.putText(
            canvas,
            f"Frame {frame_idx+1}/{len(frames)}",
            (30, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (180,180,180),
            2,
            cv2.LINE_AA
        )

        if frame.get("is_word_gap"):
            cv2.putText(
                canvas, "Next word", (CANVAS_W // 2 - 125, CANVAS_H // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (180,180,180), 2, cv2.LINE_AA,
            )
        else:
            pose = frame["pose"]
            left = frame["left_hand"]
            right = frame["right_hand"]
            draw_connections(canvas, pose, POSE_CONNECTIONS, (255,255,255), view)
            draw_landmarks(canvas, pose, (255,255,255), view)
            draw_connections(canvas, left, HAND_CONNECTIONS, (0,255,0), view)
            draw_landmarks(canvas, left, (0,255,0), view)
            draw_connections(canvas, right, HAND_CONNECTIONS, (0,0,255), view)
            draw_landmarks(canvas, right, (0,0,255), view)

        # ---------------- Per-frame label (bottom-right) ---------------- #
        frame_label = frame.get("label")
        if frame_label:
            label_text = str(frame_label)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.8
            thickness = 3
            (tw, th), _ = cv2.getTextSize(label_text, font, font_scale, thickness)
            margin = 24
            tx = CANVAS_W - tw - margin
            ty = CANVAS_H - margin
            # subtle dark backdrop for readability
            cv2.rectangle(
                canvas,
                (tx - 8, ty - th - 8),
                (tx + tw + 8, ty + 8),
                (30, 30, 30),
                -1,
            )
            cv2.putText(
                canvas,
                label_text,
                (tx, ty),
                font,
                font_scale,
                (0, 220, 180),
                thickness,
                cv2.LINE_AA,
            )

        if writer:
            writer.write(canvas)

        if not no_show:
            cv2.imshow(WINDOW_NAME, canvas)
            key = cv2.waitKey(delay) & 0xFF
            if key == ord("q"):
                if writer:
                    writer.release()
                cv2.destroyAllWindows()
                return False
            elif key == ord("r"):
                frame_idx = 0
                continue

        frame_idx += 1

    if writer:
        writer.release()

    if not no_show:
        cv2.waitKey(1000)

    print(f"Finished animating {gloss}.")
    return True


def main():
    parser = argparse.ArgumentParser(description="Animate ISL landmarks from JSON file or directory.")
    parser.add_argument("target", nargs="?", default="data/isl/keypoints/", help="Path to landmark JSON file or directory containing JSON files")
    parser.add_argument("--save-video", type=str, default=None, help="Optional output MP4 video path or directory")
    parser.add_argument("--no-show", action="store_true", help="Do not open GUI window (useful for headless video rendering)")
    args = parser.parse_args()

    target_path = Path(args.target)

    if not target_path.exists():
        print(f"Target path not found: {target_path}")
        sys.exit(1)

    if target_path.is_file():
        files = [target_path]
    else:
        files = sorted(target_path.glob("*.json"))

    if not files:
        print(f"No JSON files found in {target_path}")
        sys.exit(1)

    print(f"Found {len(files)} JSON file(s) to animate.")

    for idx, json_path in enumerate(files):
        out_video = None
        if args.save_video:
            out_dir = Path(args.save_video)
            if out_dir.is_dir() or len(files) > 1:
                out_dir.mkdir(parents=True, exist_ok=True)
                out_video = str(out_dir / f"{json_path.stem}.mp4")
            else:
                out_video = args.save_video

        cont = process_file(json_path, save_video=out_video, no_show=args.no_show)
        if not cont:
            print("Stopped by user.")
            break

    if not args.no_show:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
