import io
import json
import tempfile
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse

from src.recognition.preprocess import HandKeypointExtractor
from src.utils.feature_extractor import extract_keypoints_from_frame

app = FastAPI(title="AITE API", version="0.1.0")
extractor = HandKeypointExtractor()

@app.get("/health")
def health_check():
    return {"status": "ok", "project": "ASL-ISL Translation Engine (AITE)"}

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
    return {
        "keypoints": keypoints.tolist(),
        "message": "Recognition model not yet trained. Keypoints extracted."
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
