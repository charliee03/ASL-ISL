import time
import requests
import json
import base64

API_URL = "http://127.0.0.1:8000"
# Find any valid video in the dataset
TEST_VIDEO_PATH = "data/wlasl/videos/69205.mp4" 

def benchmark_pipeline(video_path):
    print(f"Starting E2E Benchmark on: {video_path}")
    
    print("\n[1/3] Hitting /predict-sequence (ASL Video -> ASL Gloss)...")
    start_time = time.time()
    try:
        with open(video_path, 'rb') as f:
            files = {'file': (video_path, f, 'video/mp4')}
            rec_resp = requests.post(f"{API_URL}/predict-sequence", files=files)
            rec_resp.raise_for_status()
            rec_data = rec_resp.json()
            asl_gloss = rec_data.get("gloss")
    except Exception as e:
        print(f"Error during recognition: {e}")
        return
    rec_time = time.time() - start_time
    print(f"  -> Predicted ASL Gloss: {asl_gloss}")
    print(f"  -> Time taken: {rec_time:.3f} seconds")
    
    if not asl_gloss or asl_gloss == "UNKNOWN":
        print("Using fallback 'hello' for translation...")
        asl_gloss = "hello"

    print("\n[2/3] Hitting /translate (ASL Gloss -> ISL Gloss)...")
    start_time = time.time()
    try:
        trans_resp = requests.post(f"{API_URL}/translate", json={"asl_gloss": asl_gloss})
        trans_resp.raise_for_status()
        trans_data = trans_resp.json()
        isl_gloss = trans_data.get("isl_gloss")
    except Exception as e:
        print(f"Error during translation: {e}")
        return
    trans_time = time.time() - start_time
    print(f"  -> Translated ISL Gloss: {isl_gloss}")
    print(f"  -> Time taken: {trans_time:.3f} seconds")

    print("\n[3/3] Hitting /generate-avatar (ISL Gloss -> Video)...")
    start_time = time.time()
    try:
        gen_resp = requests.post(f"{API_URL}/generate-avatar", json={"isl_gloss": isl_gloss})
        gen_resp.raise_for_status()
        gen_data = gen_resp.json()
        video_b64 = gen_data.get("video_base64")
    except Exception as e:
        print(f"Error during generation: {e}")
        return
    gen_time = time.time() - start_time
    
    print(f"  -> Time taken: {gen_time:.3f} seconds")
    
    total_time = rec_time + trans_time + gen_time
    print("\n" + "="*40)
    print("BENCHMARK SUMMARY (End-to-End Latency)")
    print("="*40)
    print(f"Recognition:    {rec_time:.3f} s")
    print(f"Translation:    {trans_time:.3f} s")
    print(f"Generation:     {gen_time:.3f} s")
    print("-" * 40)
    print(f"Total Latency:  {total_time:.3f} s")
    print("="*40)

if __name__ == "__main__":
    benchmark_pipeline(TEST_VIDEO_PATH)
