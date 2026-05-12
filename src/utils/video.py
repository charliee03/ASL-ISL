import subprocess


def extract_frames(video_path, output_pattern="frame_%04d.jpg", fps=30):
    cmd = [
        "ffmpeg", "-i", video_path,
        "-vf", f"fps={fps}",
        output_pattern
    ]
    subprocess.run(cmd, check=True)
