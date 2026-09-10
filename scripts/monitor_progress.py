import time
import os
import subprocess
import sys

TOTAL_BYTES = 56_800_000_000  # ~52.9 GB
MILESTONES = [15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]

def get_dir_size(path):
    total = 0
    try:
        for root, dirs, files in os.walk(path):
            for f in files:
                fp = os.path.join(root, f)
                total += os.path.getsize(fp)
    except Exception:
        pass
    return total

def main():
    cache_path = os.path.expanduser("~/.cache/kagglehub")
    last_reported = 10
    print(f"Started progress monitor. Current baseline ~11%. Monitoring milestones: {MILESTONES}")
    sys.stdout.flush()

    while True:
        size = get_dir_size(cache_path)
        pct = int((size / TOTAL_BYTES) * 100)
        
        for m in MILESTONES:
            if pct >= m > last_reported:
                gb = size / (1024**3)
                print(f"[MILESTONE REACHED] Download progress: {pct}% ({gb:.2f} GB / 52.9 GB)")
                sys.stdout.flush()
                last_reported = m
                break
                
        time.sleep(30)

if __name__ == "__main__":
    main()
