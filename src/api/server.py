import io
import json
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# pyrefly: ignore [missing-import]
from src.recognition.preprocess import HandKeypointExtractor    
# pyrefly: ignore [missing-import]
from src.utils.feature_extractor import extract_keypoints_from_frame
# pyrefly: ignore [missing-import]
from src.recognition.model import SignRecognitionTransformer
# pyrefly: ignore [missing-import]
from src.translation.translator import ASLtoISLTranslator

app = FastAPI(title="AITE API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

extractor = HandKeypointExtractor()
WEB_DIR = Path(__file__).resolve().parent.parent / "web"

if WEB_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")

# Load Model
NUM_FRAMES = 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = None
gloss_vocab = {}
translator = None

try:
    with open("models/recognition/gloss_vocab.json") as f:
        vocab_dict = json.load(f)

        if "id_to_gloss" in vocab_dict:
            gloss_vocab = {int(k): v for k, v in vocab_dict["id_to_gloss"].items()}
        else:
            gloss_vocab = {int(k): v for k, v in vocab_dict.items()}
    
    vocab_size = len(gloss_vocab)
    model = SignRecognitionTransformer(
        num_keypoints=27, 
        d_model=128, 
        nhead=4, 
        num_encoder_layers=3, 
        vocab_size=vocab_size,
        dropout=0.0
    )
    
    model_path = Path("models/recognition/best_model.pt")
    if model_path.exists():
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        model.to(device)
        model.eval()
        print(f"Loaded recognition model from {model_path} with {vocab_size} classes.")
    else:
        print(f"Warning: Model weights not found at {model_path}.")
        model = None
        
except Exception as e:
    print(f"Warning: Failed to load model or vocabulary. Error: {e}")

# Load Translator
try:
    translator = ASLtoISLTranslator(
        gloss_vocab_path="models/recognition/gloss_vocab.json",
        grammar_rules_path="configs/grammar_rules.json",
        config_path="configs/translation.yaml",
        quantize=True
    )
    print("✓ Translation module loaded successfully")
except Exception as e:
    print(f"Warning: Failed to load translation module. Error: {e}")


@app.get("/", response_class=HTMLResponse)
def root_page():
    index_file = WEB_DIR / "index.html"
    if index_file.exists():
        return HTMLResponse(index_file.read_text(encoding="utf-8"))
    return HTMLResponse(
        "<h1>AITE UI unavailable</h1><p>The web assets were not found.</p>",
        status_code=503,
    )

@app.get("/health")
def health_check():
    return {"status": "ok", "project": "ASL-ISL Translation Engine (AITE)", "model_loaded": model is not None}

@app.post("/extract-keypoints")
async def extract_keypoints(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        return JSONResponse({"error": "Invalid image"}, status_code=400)
    keypoints = extract_keypoints_from_frame(frame, extractor)
    return {
        "keypoints": keypoints.tolist(),
        "shape": keypoints.shape,
        "num_hands": 1 if np.any(keypoints) else 0
    }

@app.post("/predict-gloss")
async def predict_gloss(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        return JSONResponse({"error": "Invalid image"}, status_code=400)
        
    keypoints = extract_keypoints_from_frame(frame, extractor)
    
    if model is None:
        return {
            "keypoints": keypoints.tolist(),
            "message": "Recognition model not loaded."
        }
        
    # Repeat the single frame to fill the sequence length
    kps = np.expand_dims(keypoints, axis=0) # (1, 27, 3)
    kps = np.repeat(kps, NUM_FRAMES, axis=0) # (32, 27, 3)
    kps_tensor = torch.FloatTensor(kps).unsqueeze(0).to(device) # (1, 32, 27, 3)
    
    with torch.no_grad():
        outputs = model(kps_tensor)
        probs = torch.softmax(outputs, dim=1)[0]
        conf, pred_idx = torch.max(probs, dim=0)
        
    predicted_gloss = gloss_vocab.get(pred_idx.item(), "UNKNOWN")
    
    return {
        "gloss": predicted_gloss,
        "confidence": float(conf.item()),
        "keypoints": keypoints.tolist()
    }

@app.post("/translate")
async def translate_gloss(body: dict):
    """
    Translate ASL gloss to ISL gloss.
    
    Request body:
    {
        "asl_gloss": "HELLO MY NAME IS JOHN"
    }
    
    Response:
    {
        "asl_gloss": "HELLO MY NAME IS JOHN",
        "isl_gloss": "NAMASKAR MERA NAM JOHN HAI",
        "confidence": 0.85
    }
    """
    if translator is None:
        return JSONResponse(
            {"error": "Translation module not loaded"},
            status_code=500
        )
    
    asl_gloss = body.get("asl_gloss", "").strip()
    if not asl_gloss:
        return JSONResponse(
            {"error": "Missing or empty asl_gloss in request body"},
            status_code=400
        )
    
    try:
        isl_gloss = translator.translate_gloss_string(asl_gloss)
        
        return {
            "asl_gloss": asl_gloss,
            "isl_gloss": isl_gloss,
            "confidence": 0.85  # Placeholder; real confidence scoring would require reference data
        }
    except Exception as e:
        return JSONResponse(
            {"error": f"Translation failed: {str(e)}"},
            status_code=500
        )

@app.post("/predict-sequence")
async def predict_sequence(file: UploadFile = File(...)):
    if model is None:
        return JSONResponse({"error": "Model not loaded"}, status_code=500)
        
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        contents = await file.read()
        tmp.write(contents)
        tmp_path = tmp.name
        
    cap = cv2.VideoCapture(tmp_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        Path(tmp_path).unlink()
        return JSONResponse({"error": "Invalid video"}, status_code=400)
        
    indices = np.linspace(0, total_frames - 1, NUM_FRAMES, dtype=int)
    keypoints_seq = []
    
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            keypoints_seq.append(np.zeros((27, 3), dtype=np.float32))
            continue
        kp = extractor.extract(frame)
        keypoints_seq.append(kp)
        
    cap.release()
    Path(tmp_path).unlink()
    
    stacked = np.stack(keypoints_seq) # (32, 27, 3)
    kps_tensor = torch.FloatTensor(stacked).unsqueeze(0).to(device) # (1, 32, 27, 3)
    
    with torch.no_grad():
        outputs = model(kps_tensor)
        probs = torch.softmax(outputs, dim=1)[0]
        conf, pred_idx = torch.max(probs, dim=0)
        
    predicted_gloss = gloss_vocab.get(pred_idx.item(), "UNKNOWN")
    
    return {
        "gloss": predicted_gloss,
        "confidence": float(conf.item())
    }

@app.post("/generate-avatar")
async def generate_avatar_endpoint(body: dict):
    import base64
    import os
    import tempfile
    import json
    from src.generation.animate import process_file
    
    isl_gloss = body.get("isl_gloss", "").strip()
    if not isl_gloss:
        return JSONResponse({"error": "Missing isl_gloss"}, status_code=400)
        
    rev_map = {"namaskar": "hello", "alvida": "goodbye", "shukriya": "thank_you", "maafi": "sorry", "haan": "yes", "nahi": "no", "kripaya": "please", "pani": "water", "khana": "food", "madad": "help", "vyakti": "person", "nam": "name", "samay": "time", "din": "day", "raat": "night", "khush": "happy", "udaas": "sad", "pyar": "love", "dost": "friend", "parivar": "family", "main": "i", "tum": "you", "wo": "they", "ham": "we", "kaun": "who", "kya": "what", "kahan": "where", "kab": "when", "kyun": "why", "kaise": "how"}
    
    words = isl_gloss.lower().split()
    combined_frames = []
    fps = 30
    
    for word in words:
        # Check if the word is a Hindi translation that needs reversing to English filename
        mapped_word = rev_map.get(word, word)
        
        path = Path(f"data/isl/keypoints/{mapped_word}.json")
        if path.exists():
            with open(path, "r") as f:
                data = json.load(f)
                combined_frames.extend(data.get("frames", []))
                fps = data.get("fps", fps)
                
    if not combined_frames:
        # Fallback to hello
        path = Path("data/isl/keypoints/hello.json")
        if path.exists():
            with open(path, "r") as f:
                data = json.load(f)
                combined_frames.extend(data.get("frames", []))
                fps = data.get("fps", fps)
        else:
            return JSONResponse({"error": "No ISL keypoints found"}, status_code=404)
            
    # Write combined frames to temp json
    with tempfile.NamedTemporaryFile(delete=False, suffix=".json", mode="w") as tmp_json:
        json.dump({"fps": fps, "frames": combined_frames}, tmp_json)
        combined_json_path = tmp_json.name
            
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_mp4:
        out_video = tmp_mp4.name
        
    try:
        process_file(Path(combined_json_path), save_video=out_video, no_show=True, override_text=isl_gloss)
        
        web_video = out_video.replace(".mp4", "_web.mp4")
        os.system(f"ffmpeg -y -i {out_video} -vcodec libx264 -preset ultrafast -f mp4 {web_video} > /dev/null 2>&1")
        
        with open(web_video, "rb") as f:
            video_bytes = f.read()
            
        video_base64 = base64.b64encode(video_bytes).decode("utf-8")
        return {"video_base64": video_base64, "mime_type": "video/mp4"}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
    finally:
        if os.path.exists(out_video):
            os.remove(out_video)
        if 'web_video' in locals() and os.path.exists(web_video):
            os.remove(web_video)
        if os.path.exists(combined_json_path):
            os.remove(combined_json_path)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
