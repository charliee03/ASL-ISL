# MSASL Dataset: Download & Preparation Guide

This document walks through the complete procedure used to download, process, and prepare the **MS-ASL (Microsoft American Sign Language)** dataset for our ASL → ISL translation project.

The MSASL dataset contains ~25,000 annotations across 1,000 ASL sign classes, sourced from YouTube videos. Since YouTube videos can go private or get deleted over time, a significant filtering and recovery process was required.

---

## Prerequisites

Make sure you have these installed in your Python virtual environment:

```bash
pip install yt-dlp
Node.js (required for yt-dlp's JavaScript challenge solver)
```

You also need **ffmpeg** installed and available in your PATH:
- Download from [ffmpeg.org](https://ffmpeg.org/download.html) or install via `choco install ffmpeg` on Windows.

The MSASL annotation files should already be in `data/msasl/MS-ASL/`:
- `MSASL_train.json`
- `MSASL_val.json`
- `MSASL_test.json`
- `MSASL_classes.json`

---

## Step 1: Check Which YouTube Links Are Still Alive

Many of the YouTube URLs in the MSASL annotations are now dead (deleted, private, or region-locked). We first need to filter out the dead ones.

This script uses YouTube's oEmbed API to quickly check each URL's validity using 50 concurrent threads:

```python
# check_msasl_links.py (run from project root)

import json
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from collections import Counter

def check_url_validity(url):
    oembed_url = f"https://www.youtube.com/oembed?format=json&url={url}"
    try:
        req = urllib.request.Request(oembed_url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=5) as response:
            return url, response.status == 200
    except urllib.error.HTTPError:
        return url, False
    except Exception:
        return url, False

def process_datasets():
    base_dir = Path("data/msasl/MS-ASL")
    json_files = ["MSASL_train.json", "MSASL_val.json", "MSASL_test.json"]
    
    all_entries = []
    for jf in json_files:
        path = base_dir / jf
        if path.exists():
            with open(path, 'r') as f:
                all_entries.extend(json.load(f))
                
    if not all_entries:
        print("No MSASL JSON files found.")
        return
        
    unique_urls = set(entry['url'] for entry in all_entries if 'url' in entry)
    # Fix URLs missing protocol
    cleaned_urls = set()
    for u in unique_urls:
        if u.startswith("www."):
            cleaned_urls.add("https://" + u)
        else:
            cleaned_urls.add(u)
            
    print(f"Checking {len(cleaned_urls)} unique YouTube URLs...")
    
    valid_urls = set()
    invalid_urls = set()
    
    with ThreadPoolExecutor(max_workers=50) as executor:
        future_to_url = {executor.submit(check_url_validity, url): url for url in cleaned_urls}
        
        completed = 0
        for future in as_completed(future_to_url):
            url, is_valid = future.result()
            if is_valid:
                valid_urls.add(url)
            else:
                invalid_urls.add(url)
                
            completed += 1
            if completed % 500 == 0 or completed == len(cleaned_urls):
                print(f"Checked {completed}/{len(cleaned_urls)} URLs. Valid so far: {len(valid_urls)}")
                
    print(f"\nTotal valid unique URLs: {len(valid_urls)}")
    print(f"Total invalid unique URLs: {len(invalid_urls)}")
    
    # Recalculate class distributions for valid URLs
    class_counts = Counter()
    for entry in all_entries:
        u = entry.get('url', '')
        if u.startswith("www."):
            u = "https://" + u
        if u in valid_urls:
            label = entry.get('label')
            if label is not None:
                class_counts[label] += 1
                
    counts = list(class_counts.values())
    print(f"\n[VALID CLASS DISTRIBUTION]")
    print(f"Unique classes remaining: {len(class_counts)}")
    if counts:
        print(f"Videos per class -> Min: {min(counts)}, Max: {max(counts)}, Mean: {sum(counts)/len(counts):.1f}")
        
        weak = sum(1 for c in counts if c < 10)
        medium = sum(1 for c in counts if 10 <= c < 30)
        strong = sum(1 for c in counts if c >= 30)
        
        print(f"\nClass Strength:")
        print(f" - Weak (< 10 videos):    {weak} classes")
        print(f" - Medium (10-29 videos): {medium} classes")
        print(f" - Strong (30+ videos):   {strong} classes")
        
    # Write valid URLs to file
    with open("valid_msasl_links.txt", "w") as f:
        for url in valid_urls:
            f.write(f"{url}\n")
    print(f"\nWrote valid URLs to 'valid_msasl_links.txt'")

if __name__ == "__main__":
    process_datasets()
```

**Run it:**
```bash
python check_msasl_links.py
```

**Our results:**
- Out of ~7,300 unique YouTube URLs, **4,216** were still alive.
- This gave us **18,032** valid annotations across all splits.
- **833 classes** had ≥10 valid videos (the rest were too weak to train on).
> **Your numbers may differ because YouTube availability changes over time.**

The output file `valid_msasl_links.txt` was saved to the root directory and is used by all subsequent scripts.

---

## Step 2: Download Video Clips Using yt-dlp

With the valid links identified, we download only the specific time segments we need from each YouTube video using `yt-dlp`'s `--download-sections` feature.

Each annotation in MSASL has `start_time` and `end_time` fields that mark exactly where the sign occurs in the YouTube video. We download only that segment, not the entire video.

```python
# scripts/download_videos.py

import json
import subprocess
from collections import Counter
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent
MSASL_DIR = BASE_DIR.parent / "data" / "msasl" / "MS-ASL"
VIDEOS_DIR = MSASL_DIR / "videos"
TRAIN_JSON = MSASL_DIR / "MSASL_train.json"
VAL_JSON = MSASL_DIR / "MSASL_val.json"
TEST_JSON = MSASL_DIR / "MSASL_test.json"
VALID_LINKS_FILE = BASE_DIR.parent / "valid_msasl_links.txt"

def get_valid_links():
    valid_links = set()
    if VALID_LINKS_FILE.exists():
        with open(VALID_LINKS_FILE, "r") as f:
            for line in f:
                valid_links.add(line.strip())
        print(f"Loaded {len(valid_links)} valid URLs from {VALID_LINKS_FILE.name}")
    return valid_links

def load_annotations():
    all_data = []
    for json_file in [TRAIN_JSON, VAL_JSON, TEST_JSON]:
        if json_file.exists():
            with open(json_file, 'r') as f:
                all_data.extend(json.load(f))
    return all_data

def download_video(url, start_time, end_time, output_path):
    cmd = [
        "yt-dlp",
        "-f", "best[ext=mp4]/best",
        "--download-sections", f"*{start_time}-{end_time}",
        "--force-keyframes-at-cuts",
        "--retries", "3",
        "--no-warnings",
        "--quiet",
        "-o", str(output_path),
        url
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        return True
    except subprocess.CalledProcessError:
        return False
    except FileNotFoundError:
        print("ERROR: 'yt-dlp' is not installed or not in your PATH.")
        exit(1)

def main():
    VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Verified output directory: {VIDEOS_DIR}")
    
    valid_links = get_valid_links()
    annotations = load_annotations()
    
    if not annotations:
        print("No annotations found.")
        return
    
    # Build download list
    to_download = []
    for i, entry in enumerate(annotations):
        url = entry.get('url', '')
        if url.startswith('www.'):
            url = 'https://' + url
            
        label = entry.get('label')
        
        if valid_links and url not in valid_links:
            continue
            
        start_time = entry.get('start_time', 0.0)
        end_time = entry.get('end_time', 0.0)
        
        if end_time <= start_time:
            end_time = start_time + 3.0
        
        signer_id = entry.get('signer_id', 'X')
        output_filename = f"class_{label}_signer_{signer_id}_{i}.mp4"
        output_path = VIDEOS_DIR / output_filename
        
        if not output_path.exists():
            to_download.append({
                'url': url,
                'start_time': start_time,
                'end_time': end_time,
                'output_path': output_path,
                'label': label
            })
            
    print(f"Found {len(to_download)} clips to download.")
    
    if len(to_download) == 0:
        print("Everything is already downloaded!")
        return
        
    success_count = 0
    fail_count = 0
    
    print("\nStarting downloads...")
    for idx, item in enumerate(to_download, 1):
        print(f"[{idx}/{len(to_download)}] Downloading Class {item['label']}... ", end="", flush=True)
        
        if download_video(item['url'], item['start_time'], item['end_time'], item['output_path']):
            print("SUCCESS")
            success_count += 1
        else:
            print("FAILED")
            fail_count += 1
            
    print(f"\nDownload Complete! Success: {success_count}, Failed: {fail_count}")

if __name__ == "__main__":
    main()
```

**Run it:**
```bash
python -m scripts.download_videos
```

> **Important notes:**
> - YouTube rate-limits automated downloads. We ran this script **multiple times** (7 runs in our case). Each re-run skips already-downloaded files automatically.
> - Expect a significant failure rate (~40–60%) due to bot detection and rate limiting. This is normal — that's what Step 3 is for.
> - You can increase concurrency by using `ThreadPoolExecutor`, but this makes rate-limiting worse. We found 8 workers to be a good balance between speed and not getting blocked too often.

---

## Step 3: Download Full Videos for Failed Clips (Manual)

After multiple runs of the download script, a large number of clips still fail because YouTube's bot detection blocks `yt-dlp` for those specific videos. The workaround is to **manually download the full YouTube videos** and then extract the clips locally using `ffmpeg`.

### 3a. Build the list of failed URLs

The download script from Step 2 logs every failed clip to a text file (`downloaded_videos.txt`). We need to extract the unique YouTube URLs from all the failures, so we know which full videos to download.

Create a file called `url_list.py` (in the same directory where you'll run the manual download script) containing a simple Python list of the unique failed URLs:

```python
# url_list.py
# Paste the unique YouTube URLs that failed during Step 2 here.
# You can extract these manually from downloaded_videos.txt, or write a
# quick script to parse lines containing "FAILED (DOWNLOAD_FAILED)" and
# pull out the URL after "URL: ".

urls = [
    "https://www.youtube.com/watch?v=RffIEzlN5Yo",
    "https://www.youtube.com/watch?v=WxY7E9P46DM",
    "https://www.youtube.com/watch?v=TYEae1fcehg",
    # ... add all unique failed URLs here
]
```

> **Tip:** You can automate building this list. For example, this one-liner extracts unique URLs from `downloaded_videos.txt`:
> ```python
> import re
> with open("downloaded_videos.txt") as f:
>     urls = list(set(re.findall(r'FAILED \(DOWNLOAD_FAILED\) - URL: (https://www\.youtube\.com/watch\?v=[^\s]+)', f.read())))
> print(f"Found {len(urls)} unique failed URLs")
> with open("url_list.py", "w") as out:
>     out.write("urls = [\n")
>     for u in sorted(urls):
>         out.write(f'    "{u}",\n')
>     out.write("]\n")
> ```

### 3b. Export browser cookies for authenticated downloads

YouTube's bot detection blocks many `yt-dlp` requests. To bypass this, we export cookies from a logged-in browser session and pass them to `yt-dlp`. This makes YouTube treat the download as a normal browser visit.

**How to export cookies:**
1. Install a browser extension like **"Get cookies.txt LOCALLY"** (available for Chrome/Edge).
2. Open YouTube in your browser and make sure you're logged in.
3. Click the extension and export cookies as a `.txt` file (Netscape format).
4. Save it as `cookies.txt` in the directory where you'll run the download script.

> **Using multiple cookie files:** YouTube may rate-limit a single account. To work around this, export cookies from multiple browser profiles (or accounts) and save them as separate files (e.g., `cookies_1.txt`, `cookies_2.txt`, `cookies_3.txt`). The script below splits the URL list across these cookie files and downloads in parallel, one thread per cookie file.

### 3c. Download full videos using cookies

This script downloads the **complete** YouTube video (not just a clip) for each failed URL. It uses multiple cookie files in parallel to avoid rate limits, validates each download with `ffprobe`, and automatically retries failures with a different cookie file.

Save the full videos in a dedicated folder (e.g., `missed_videos\`). Each video is saved as `{video_id}.mp4`.

```python
# download_full_videos.py
# Run from any directory. Requires: yt-dlp, ffmpeg/ffprobe, Node.js (for JS challenges)

from pathlib import Path
from urllib.parse import urlparse, parse_qs
from yt_dlp import YoutubeDL
import subprocess
import time
import random
from url_list import urls  # <-- The list of failed URLs from Step 3a
from concurrent.futures import ThreadPoolExecutor
import math
from threading import Lock

counter_lock = Lock()

# CONFIGURATION — Change these to match your setup

# Where to save the full downloaded videos
DOWNLOAD_DIR = Path("missed_videos") # edit it if you want to save them elsewhere
DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Cookie files exported from your browser (one per profile/account).
# Add as many as you have. The script will use one per parallel worker.
COOKIE_FILES = []  # each cookie file's path to be added properly into the list

NUM_WORKERS = len(COOKIE_FILES)  # One download thread per cookie file

# HELPER FUNCTIONS
RED = "\033[31m"
YELLOW = "\033[33m"
GREEN = "\033[32m"
ORANGE = "\033[38;5;214m"
RESET = "\033[0m"

def get_video_id(url):
    """Extract the YouTube video ID from a URL."""
    return parse_qs(urlparse(url).query)["v"][0]

def is_valid_video(video_path):
    """
    Check if a video file exists and contains a valid video stream.
    Uses ffprobe for verification; falls back to file-size check.
    """
    if not video_path.exists():
        return False
    if video_path.stat().st_size < 10 * 1024:  # Reject files < 10 KB
        return False
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=codec_name",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(video_path),
            ],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        return result.returncode == 0 and result.stdout.strip() != ""
    except FileNotFoundError:
        return video_path.stat().st_size > 1024

# DOWNLOAD LOGIC
total_videos = len(urls)
downloaded_count = 0
skipped_count = 0

def download_urls(url_list, cookie_file, failed_urls):
    """Download a list of URLs using a specific cookie file."""
    global downloaded_count, skipped_count

    for url in url_list:
        video_id = get_video_id(url)
        output_file = DOWNLOAD_DIR / f"{video_id}.mp4"

        # Skip if already downloaded and valid
        if is_valid_video(output_file):
            print(f"{ORANGE}✓ {video_id}.mp4 already exists. Skipping.{RESET}\n")
            with counter_lock:
                skipped_count += 1
            continue

        # Remove corrupt/incomplete file if present
        if output_file.exists():
            print(f"{RED}Removing invalid file: {output_file.name}{RESET}")
            output_file.unlink()

        print(f"Downloading {video_id} with {cookie_file}...")

        ydl_opts = {
            "format": "best[ext=mp4][height<=720]/bv*[height<=720]+ba/best",
            "merge_output_format": "mp4",
            "outtmpl": str(DOWNLOAD_DIR / f"{video_id}.%(ext)s"),
            "cookiefile": cookie_file,
            "noplaylist": True,
            "ignoreerrors": True,
            "sleep_interval": 2,
            "max_sleep_interval": 5,
            "retries": 5,
            "fragment_retries": 5,
            "socket_timeout": 30,
            "postprocessor_args": ["-movflags", "+faststart"],
        }

        try:
            with YoutubeDL(ydl_opts) as ydl:
                ret = ydl.download([url])

            if is_valid_video(output_file):
                print(f"{GREEN}✓ Successfully downloaded {video_id}.mp4{RESET}\n")
                with counter_lock:
                    downloaded_count += 1
            else:
                print(f"{RED}✗ Download failed or file is invalid: {video_id}.mp4{RESET}\n")
                failed_urls.append(url)
                if output_file.exists():
                    output_file.unlink()

        except KeyboardInterrupt:
            print(f"{YELLOW}\nDownload interrupted by user.{RESET}")
            break
        except Exception as e:
            print(f"{RED}✗ Failed to download {video_id}: {e}{RESET}\n")
            failed_urls.append(url)
        finally:
            time.sleep(random.uniform(5, 10))

# PARALLEL EXECUTION — Split URLs across cookie files
chunk_size = math.ceil(len(urls) / NUM_WORKERS)
parts = [urls[i * chunk_size : (i + 1) * chunk_size] for i in range(NUM_WORKERS)]
failed_lists = [[] for _ in range(NUM_WORKERS)]

print(f"Starting parallel download of {total_videos} videos "
      f"across {NUM_WORKERS} workers...\n")

with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
    futures = [
        executor.submit(download_urls, parts[i], COOKIE_FILES[i], failed_lists[i])
        for i in range(NUM_WORKERS)
    ]
    for future in futures:
        future.result()

# RETRY — Try failed URLs with a different cookie file
retry_urls = list(dict.fromkeys(url for failed in failed_lists for url in failed))

if retry_urls:
    print(f"{YELLOW}\nRetrying {len(retry_urls)} failed videos...{RESET}\n")
    final_failed = []

    for url in retry_urls:
        # Figure out which cookie originally failed, retry with the others
        original_idx = next((i for i, fl in enumerate(failed_lists) if url in fl), 0)
        retry_cookies = [COOKIE_FILES[i] for i in range(NUM_WORKERS) if i != original_idx]

        success = False
        for cookie in retry_cookies:
            video_id = get_video_id(url)
            output_file = DOWNLOAD_DIR / f"{video_id}.mp4"

            if is_valid_video(output_file):
                print(f"{GREEN}✓ {video_id}.mp4 already exists on retry.{RESET}\n")
                with counter_lock:
                    downloaded_count += 1
                success = True
                break

            if output_file.exists():
                output_file.unlink()

            print(f"Retrying {video_id} with {cookie}...")
            ydl_opts = {
                "format": "best[ext=mp4][height<=720]/bv*[height<=720]+ba/best",
                "merge_output_format": "mp4",
                "outtmpl": str(DOWNLOAD_DIR / f"{video_id}.%(ext)s"),
                "cookiefile": cookie,
                "noplaylist": True,
                "ignoreerrors": True,
                "sleep_interval": 2,
                "max_sleep_interval": 5,
                "retries": 5,
                "fragment_retries": 5,
                "socket_timeout": 30,
                "postprocessor_args": ["-movflags", "+faststart"],
            }
            try:
                with YoutubeDL(ydl_opts) as ydl:
                    ydl.download([url])

                if is_valid_video(output_file):
                    print(f"{GREEN}✓ Successfully downloaded "
                          f"{video_id}.mp4 on retry{RESET}\n")
                    with counter_lock:
                        downloaded_count += 1
                    success = True
                    break
                else:
                    if output_file.exists():
                        output_file.unlink()

            except KeyboardInterrupt:
                print(f"{YELLOW}\nInterrupted.{RESET}")
                break
            except Exception as e:
                print(f"{RED}✗ Retry failed for {video_id}: {e}{RESET}\n")
            finally:
                time.sleep(random.uniform(3, 7))

        if not success:
            final_failed.append(url)

    failed_count = len(final_failed)
    failed_videos = [f"{get_video_id(u)}.mp4" for u in final_failed]
else:
    failed_count = 0
    failed_videos = []

# SUMMARY
print("\n" + "=" * 70)
print("DOWNLOAD SUMMARY")
print("=" * 70)
print(f"Total videos in URL list      : {total_videos}")
print(f"Already downloaded (Skipped)  : {skipped_count}")
print(f"Successfully downloaded       : {downloaded_count}")
print(f"Failed downloads              : {failed_count}")
remaining = total_videos - skipped_count - downloaded_count
print(f"Remaining (not obtained)      : {remaining}")

if failed_videos:
    print("\nFailed video files:")
    for filename in failed_videos:
        print(f"  - {filename}")
else:
    print(f"\n{GREEN}No failed downloads!{RESET}")
print("=" * 70)
```

**Run it:**
```bash
python download_full_videos.py
```

> **How this script works:**
> - It splits the URL list evenly across `N` cookie files and downloads them in parallel (one thread per cookie).
> - Each downloaded video is validated using `ffprobe` to catch corrupted files.
> - Any URL that fails on its original cookie file is automatically retried using one of the other cookie files.
> - Videos are saved as `{video_id}.mp4` (e.g., `RffIEzlN5Yo.mp4`) in the download directory.
> - The script sleeps 5–10 seconds between downloads to avoid triggering rate limits.

### 3d. Extract clips from the manually downloaded videos

Once you have the full videos saved in a folder (we used `missed_videos\`), this script goes through all the MSASL annotations and extracts the specific time segments using `ffmpeg`:

```python
# scripts/extract_from_missed.py

import json
import subprocess
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent
MSASL_DIR = BASE_DIR.parent / "data" / "msasl" / "MS-ASL"
VIDEOS_DIR = MSASL_DIR / "videos"
TRAIN_JSON = MSASL_DIR / "MSASL_train.json"
VAL_JSON = MSASL_DIR / "MSASL_val.json"
TEST_JSON = MSASL_DIR / "MSASL_test.json"
VALID_LINKS_FILE = BASE_DIR.parent / "valid_msasl_links.txt"
MISSED_DIR = Path(r"missed_videos")  # <-- Change this to wherever you saved the full videos

def get_valid_links():
    valid_links = set()
    if VALID_LINKS_FILE.exists():
        with open(VALID_LINKS_FILE, "r") as f:
            for line in f:
                valid_links.add(line.strip())
        print(f"Loaded {len(valid_links)} valid URLs from {VALID_LINKS_FILE.name}")
    return valid_links

def load_annotations():
    all_data = []
    for json_file in [TRAIN_JSON, VAL_JSON, TEST_JSON]:
        if json_file.exists():
            with open(json_file, 'r') as f:
                all_data.extend(json.load(f))
    return all_data

def extract_video_id(url):
    if "v=" in url:
        return url.split("v=")[1].split("&")[0]
    elif "youtu.be/" in url:
        return url.split("youtu.be/")[1].split("?")[0]
    return url.split("/")[-1]

def extract_clip(full_video_path, start_time, end_time, output_path):
    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(full_video_path),
        "-ss", str(start_time),
        "-to", str(end_time),
        "-c:v", "libx264",
        "-c:a", "aac",
        "-preset", "fast",
        "-loglevel", "error",
        str(output_path)
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except subprocess.CalledProcessError:
        if output_path.exists():
            output_path.unlink()
        return False

def main():
    VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
    
    valid_links = get_valid_links()
    annotations = load_annotations()
    
    if not annotations:
        print("No annotations found.")
        return

    clips_extracted = 0
    clips_already_present = 0
    missing_urls = set()
    
    print(f"Processing {len(annotations)} annotations...")
    
    for i, entry in enumerate(annotations):
        url = entry.get('url', '')
        if url.startswith('www.'):
            url = 'https://' + url
            
        if valid_links and url not in valid_links:
            continue
            
        label = entry.get('label')
        start_time = entry.get('start_time', 0.0)
        end_time = entry.get('end_time', 0.0)
        
        if end_time <= start_time:
            end_time = start_time + 3.0
            
        signer_id = entry.get('signer_id', 'X')
        output_filename = f"class_{label}_signer_{signer_id}_{i}.mp4"
        output_path = VIDEOS_DIR / output_filename
        
        if output_path.exists():
            clips_already_present += 1
            continue
            
        video_id = extract_video_id(url)
        full_video_path = MISSED_DIR / f"{video_id}.mp4"
        
        if full_video_path.exists():
            print(f"[{i+1}/{len(annotations)}] Extracting {output_filename}...")
            if extract_clip(full_video_path, start_time, end_time, output_path):
                clips_extracted += 1
        else:
            missing_urls.add(url)
            
    print("\n--- Summary ---")
    print(f"Clips extracted from missed: {clips_extracted}")
    print(f"Clips already present: {clips_already_present}")
    print(f"Full videos still missing: {len(missing_urls)}")

if __name__ == "__main__":
    main()
```

**Run it:**
```bash
python -m scripts.extract_from_missed
```

> **How the naming convention works:**
> - The full videos in the `missed/` folder must be named by their YouTube video ID (e.g., `RffIEzlN5Yo.mp4`).
> - The script automatically matches each annotation's URL to the correct full video file and extracts the relevant segment.

---

## Step 4: Create the Unified Annotation File

After all clips are downloaded and extracted, we create a single JSON file (`MSASL_unified.json`) that maps every video clip to its label, split (train/val/test), and class name. This file is what the training pipeline reads.

```python
# scripts/create_unified_annotation.py

import json
from pathlib import Path
from collections import Counter

# Paths
BASE_DIR = Path(__file__).parent
MSASL_DIR = BASE_DIR.parent / "data" / "msasl" / "MS-ASL"
VIDEOS_DIR = MSASL_DIR / "videos"
TRAIN_JSON = MSASL_DIR / "MSASL_train.json"
VAL_JSON = MSASL_DIR / "MSASL_val.json"
TEST_JSON = MSASL_DIR / "MSASL_test.json"
CLASSES_JSON = MSASL_DIR / "MSASL_classes.json"
VALID_LINKS_FILE = BASE_DIR.parent / "valid_msasl_links.txt"
UNIFIED_JSON_PATH = MSASL_DIR / "MSASL_unified.json"

def create_unified():
    print("Creating unified annotation JSON...\n")

    # 1. Load valid links
    valid_links = set()
    with open(VALID_LINKS_FILE, 'r') as f:
        valid_links = set(line.strip() for line in f)
    print(f"Loaded {len(valid_links)} valid URLs.")

    # 2. Load class names (label ID -> gloss name)
    with open(CLASSES_JSON, 'r') as f:
        classes = json.load(f)
    class_map = {i: gloss for i, gloss in enumerate(classes)}

    # 3. Load all annotations, tagging each with its split
    splits = [("train", TRAIN_JSON), ("val", VAL_JSON), ("test", TEST_JSON)]
    all_annotations = []
    for split_name, json_path in splits:
        with open(json_path, 'r') as f:
            data = json.load(f)
            for entry in data:
                entry['split'] = split_name
            all_annotations.extend(data)
    print(f"Loaded {len(all_annotations)} total annotations.")

    # 4. Filter by valid links
    valid_annotations = []
    for i, entry in enumerate(all_annotations):
        url = entry.get('url', '')
        if url.startswith('www.'):
            url = 'https://' + url
        if url in valid_links:
            entry['original_index'] = i
            valid_annotations.append(entry)

    # 5. Only keep classes with >= 10 valid videos
    class_counts = Counter(entry.get('label') for entry in valid_annotations)
    valid_classes = {label for label, count in class_counts.items() if count >= 10}
    filtered = [e for e in valid_annotations if e.get('label') in valid_classes]

    # 6. Build unified JSON (only include entries whose video file actually exists)
    unified_data = []
    for entry in filtered:
        label = entry.get('label')
        signer_id = entry.get('signer_id', 'X')
        i = entry['original_index']
        filename = f"class_{label}_signer_{signer_id}_{i}.mp4"

        if (VIDEOS_DIR / filename).exists():
            unified_data.append({
                "video": f"videos/{filename}",
                "gloss": class_map.get(label, f"class_{label}"),
                "split": entry["split"],
                "label": label,
                "signer_id": signer_id,
                "start_time": entry.get("start_time", 0.0),
                "end_time": entry.get("end_time", 0.0),
                "box": entry.get("box", [])
            })

    # 7. Save
    with open(UNIFIED_JSON_PATH, 'w') as f:
        json.dump(unified_data, f, indent=2)

    print(f"\nSaved: {UNIFIED_JSON_PATH}")
    print(f"Total entries: {len(unified_data)}")
    split_counts = Counter(e["split"] for e in unified_data)
    for split, count in split_counts.items():
        print(f"  {split}: {count}")

if __name__ == '__main__':
    create_unified()
```

**Run it:**
```bash
python -m scripts.create_unified_annotation
```

**Our results:**
```
Total entries: 16785
  train: 11547
  val: 2653
  test: 2585
```

---

## Step 5: Verify Everything

Run the verification script to make sure all pieces are in place:

```python
# scripts/verify_msasl_setup.py

import json
from pathlib import Path
from collections import Counter

# Change to your path if they are different than this.
BASE_DIR = Path(__file__).parent
MSASL_DIR = BASE_DIR.parent / "data" / "msasl" / "MS-ASL"
VIDEOS_DIR = MSASL_DIR / "videos"
TRAIN_JSON = MSASL_DIR / "MSASL_train.json"
VAL_JSON = MSASL_DIR / "MSASL_val.json"
TEST_JSON = MSASL_DIR / "MSASL_test.json"
VALID_LINKS_FILE = BASE_DIR.parent / "valid_msasl_links.txt"
UNIFIED_JSON_PATH = MSASL_DIR / "MSASL_unified.json"

def verify():
    print("=" * 53)
    print("      MSASL DATASET VERIFICATION")
    print("=" * 53 + "\n")
    
    # Load valid links
    valid_links = set()
    with open(VALID_LINKS_FILE, 'r') as f:
        valid_links = set(line.strip() for line in f)
    print(f"[OK] Found valid_msasl_links.txt with {len(valid_links)} URLs.")
        
    # Load annotations
    splits = [("train", TRAIN_JSON), ("val", VAL_JSON), ("test", TEST_JSON)]
    all_annotations = []
    for split_name, json_path in splits:
        with open(json_path, 'r') as f:
            data = json.load(f)
            for entry in data:
                entry['split'] = split_name
            all_annotations.extend(data)
    print(f"[OK] Loaded {len(all_annotations)} total annotations.")
    
    # Filter by valid links
    valid_annotations = []
    for i, entry in enumerate(all_annotations):
        url = entry.get('url', '')
        if url.startswith('www.'):
            url = 'https://' + url
        if url in valid_links:
            entry['original_index'] = i
            valid_annotations.append(entry)
    print(f"     -> {len(valid_annotations)} annotations have valid URLs.")
    
    # Filter classes >= 10 videos
    class_counts = Counter(entry.get('label') for entry in valid_annotations)
    valid_classes = {label for label, count in class_counts.items() if count >= 10}
    filtered = [e for e in valid_annotations if e.get('label') in valid_classes]
    print(f"[OK] {len(valid_classes)} classes with >= 10 videos ({len(filtered)} clips).\n")
    
    # Check downloaded clips
    print("--- Checking Downloaded Clips ---")
    missing = []
    found = 0
    for entry in filtered:
        label = entry.get('label')
        signer_id = entry.get('signer_id', 'X')
        filename = f"class_{label}_signer_{signer_id}_{entry['original_index']}.mp4"
        if (VIDEOS_DIR / filename).exists():
            found += 1
        else:
            missing.append(filename)
    
    print(f"Clips found: {found} / {len(filtered)}")
    if missing:
        print(f"[FAIL] Missing {len(missing)} clips!")
        for clip in missing[:5]:
            print(f"       - {clip}")
        if len(missing) > 5:
            print(f"       ... and {len(missing) - 5} more.")
    else:
        print("[OK] All required clips are present!")
    
    # Check unified annotation
    print("\n--- Checking Unified Annotation File ---")
    if UNIFIED_JSON_PATH.exists():
        with open(UNIFIED_JSON_PATH, 'r') as f:
            unified = json.load(f)
        print(f"[OK] MSASL_unified.json has {len(unified)} entries.")
    else:
        print("[FAIL] MSASL_unified.json not found! Run the create_unified_annotation script first.")
    
    # Summary
    print("\n" + "=" * 53)
    print("                  SUMMARY")
    print("=" * 53)
    
    downloads_ok = len(missing) == 0
    annotation_ok = UNIFIED_JSON_PATH.exists()
    
    print(f"{'✅' if downloads_ok else '❌'} Downloaded Clips: {'Complete' if downloads_ok else 'INCOMPLETE'}")
    print(f"{'✅' if annotation_ok else '❌'} Unified Annotation: {'Complete' if annotation_ok else 'INCOMPLETE'}")
    
    if downloads_ok and annotation_ok:
        print("\n🎉 MSASL dataset is fully set up and ready for training!")
    else:
        print("\n⚠️  Some steps are still incomplete. Check the output above.")

if __name__ == '__main__':
    verify()
```

**Run it:**
```bash
python -m scripts.verify_msasl_setup
```

**Expected output:**
```
✅ Downloaded Clips: Complete
✅ Unified Annotation: Complete

🎉 MSASL dataset is fully set up and ready for training!
```


---

### File structure after completion

```
data/msasl/MS-ASL/
├── MSASL_train.json          # Original train annotations
├── MSASL_val.json            # Original val annotations
├── MSASL_test.json           # Original test annotations
├── MSASL_classes.json        # 1000 class names (index -> gloss)
├── MSASL_unified.json        # ← Generated in Step 4 (what the model reads)
└── videos/
    ├── class_0_signer_0_3287.mp4
    ├── class_0_signer_101_24734.mp4
    ├── ...
    └── class_999_signer_95_9264.mp4   (~18,000 .mp4 files)
```

### Naming convention

Each video clip is named: `class_{label}_signer_{signer_id}_{index}.mp4`

- `label` — the integer class ID (0–999), maps to a sign word via `MSASL_classes.json`
- `signer_id` — identifies the person signing in the video
- `index` — the annotation's position in the combined train+val+test list (this ensures unique filenames even when the same signer performs the same sign multiple times)

### Notes

- The `videos/` folder contains ~18,000 files (for all valid URLs), but the unified annotation only references ~16,785 of them (the ones belonging to the 833 classes with ≥10 videos). The extra files are harmless and are simply ignored by the training pipeline.
- Spatial bounding box cropping (using the `box` field in annotations) was **not** applied during download. The keypoint extractor (MediaPipe) handles person detection at inference time, so full-frame videos work fine.
