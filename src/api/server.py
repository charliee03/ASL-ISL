import io
import json
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# pyrefly: ignore [missing-import]
from src.recognition.preprocess import HandKeypointExtractor    
# pyrefly: ignore [missing-import]
from src.utils.feature_extractor import extract_keypoints_from_frame
# pyrefly: ignore [missing-import]
from src.recognition.model import SignRecognitionTransformer

app = FastAPI(title="AITE API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

extractor = HandKeypointExtractor()

# Load Model
NUM_FRAMES = 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = None
gloss_vocab = {}

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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
