import json
import os
import tempfile
import time
import uuid
import subprocess
from difflib import SequenceMatcher
from pathlib import Path

import cv2
import numpy as np
import torch
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# pyrefly: ignore [missing-import]
from src.recognition.model import SignRecognitionTransformer
# pyrefly: ignore [missing-import]
from src.translation.translator import ASLtoISLTranslator
# pyrefly: ignore [missing-import]
from src.generation.animate import process_file

app = FastAPI(title="AITE API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

PROJECT_DIR = Path(__file__).resolve().parents[2]
WEB_DIR = Path(__file__).resolve().parent.parent / "web"
GENERATED_DIR = Path(os.getenv("AITE_GENERATED_DIR", "/tmp/aite-generated"))
LANDMARK_DIR = PROJECT_DIR / "data" / "isl" / "keypoints"
CSLRT_LANDMARK_DIR = PROJECT_DIR / "data" / "isl" / "cslrt_keypoints"
GENERATED_DIR.mkdir(parents=True, exist_ok=True)

if WEB_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")
app.mount("/generated", StaticFiles(directory=str(GENERATED_DIR)), name="generated")


@app.get("/", include_in_schema=False)
def serve_web_app():
    """Serve the browser client while keeping its assets under `/static`."""
    index_path = WEB_DIR / "index.html"
    if not index_path.is_file():
        return JSONResponse(status_code=500, content={"detail": "Web client is not installed."})
    return FileResponse(index_path, media_type="text/html")

# Load Model
NUM_FRAMES = 32
RECOGNITION_ENABLED = os.getenv("AITE_ENABLE_RECOGNITION", "false").lower() == "true"
RECOGNITION_CHECKPOINT = Path(
    os.getenv(
        "AITE_RECOGNITION_CHECKPOINT",
        str(PROJECT_DIR / "models/msasl100_pose_balanced_large_recognition/best_model.pt"),
    )
)
_confidence_override = os.getenv("AITE_RECOGNITION_MIN_CONFIDENCE")
RECOGNITION_MIN_CONFIDENCE = float(_confidence_override) if _confidence_override else 0.50
RECOGNITION_THRESHOLD_SOURCE = "environment" if _confidence_override else "uncalibrated_default"
MAX_UPLOAD_BYTES = int(os.getenv("AITE_MAX_UPLOAD_BYTES", str(50 * 1024 * 1024)))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = None
gloss_vocab = {}
recognition_model_config = {}

AVATAR_GLOSS_MAP_PATH = PROJECT_DIR / "configs" / "avatar_gloss_map.json"


def _load_landmark_assets() -> dict[str, list[Path]]:
    metadata_path = LANDMARK_DIR / "metadata.json"
    if not metadata_path.exists():
        return {}
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    assets: dict[str, list[Path]] = {}
    for entry in metadata:
        gloss = str(entry.get("gloss", "")).lower()
        path = LANDMARK_DIR / str(entry.get("landmarks", ""))
        if gloss and path.is_file():
            assets.setdefault(gloss, []).append(path)
    return assets


LANDMARK_ASSETS = _load_landmark_assets()


def _normalise_sentence_lookup(value: str) -> str:
    """Make exact corpus-sentence lookup tolerant of case and punctuation."""
    return " ".join("".join(char if char.isalnum() else " " for char in value.lower()).split())


def _load_cslrt_sentence_assets() -> dict[str, dict]:
    """Load the best tracked recorded pose sequence for each CSLRT sentence.

    These are retrieval assets, not output from a pose-generation model.
    """
    metadata_path = CSLRT_LANDMARK_DIR / "metadata.json"
    if not metadata_path.exists():
        return {}
    try:
        entries = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    assets: dict[str, dict] = {}
    for entry in entries:
        sentence = str(entry.get("sentence", "")).strip()
        asset_path = CSLRT_LANDMARK_DIR / str(entry.get("landmarks", ""))
        key = _normalise_sentence_lookup(sentence)
        if not key or not asset_path.is_file():
            continue
        candidate = {
            "path": asset_path,
            "sentence": sentence,
            "signer": str(entry.get("signer", "unknown")),
            "hand_coverage": float(entry.get("hand_coverage", 0.0)),
        }
        # Prefer the recording whose hands were detected in the most frames.
        if key not in assets or candidate["hand_coverage"] > assets[key]["hand_coverage"]:
            assets[key] = candidate
    return assets


CSLRT_SENTENCE_ASSETS = _load_cslrt_sentence_assets()


def _find_cslrt_sentence_asset(value: str, allow_approximate: bool = False) -> tuple[dict | None, str | None, float | None]:
    key = _normalise_sentence_lookup(value)
    if not key:
        return None, None, None
    exact = CSLRT_SENTENCE_ASSETS.get(key)
    if exact:
        return exact, "exact", 1.0
    if not allow_approximate:
        return None, None, None

    input_tokens = set(key.split())
    input_negated = bool(input_tokens & {"no", "not", "never"})
    best_asset, best_score = None, 0.0
    for sentence_key, asset in CSLRT_SENTENCE_ASSETS.items():
        sentence_tokens = set(sentence_key.split())
        if input_negated != bool(sentence_tokens & {"no", "not", "never"}):
            continue
        token_overlap = len(input_tokens & sentence_tokens) / max(len(input_tokens | sentence_tokens), 1)
        sequence_score = SequenceMatcher(None, key, sentence_key).ratio()
        if token_overlap >= 0.70 and sequence_score > best_score:
            best_asset, best_score = asset, sequence_score
    if best_asset is not None and best_score >= 0.88:
        return best_asset, "approximate", best_score
    return None, None, None


def _load_avatar_aliases() -> dict[str, str]:
    if not AVATAR_GLOSS_MAP_PATH.exists():
        return {}
    try:
        aliases = json.loads(AVATAR_GLOSS_MAP_PATH.read_text(encoding="utf-8")).get("aliases", {})
        return {str(alias).lower(): str(target).lower() for alias, target in aliases.items()}
    except (OSError, json.JSONDecodeError):
        return {}


AVATAR_ALIASES = _load_avatar_aliases()


translator = None

try:
    translator = ASLtoISLTranslator(
        grammar_rules_path=str(PROJECT_DIR / "configs" / "grammar_rules.json"),
        config_path=str(PROJECT_DIR / "configs" / "translation.yaml"),
        enable_llm=os.getenv("AITE_ENABLE_LLM", "false").lower() == "true",
    )
except Exception as error:
    # Translation remains available as a clearly reported API error rather than
    # preventing the web service from starting if its optional assets are absent.
    print(f"Translation unavailable: {error}")

try:
    if not RECOGNITION_ENABLED:
        raise RuntimeError("Recognition is disabled pending validated model deployment")
    checkpoint = torch.load(RECOGNITION_CHECKPOINT, map_location=device, weights_only=False)
    classes = checkpoint["classes"]
    recognition_model_config = checkpoint["model_config"]
    if checkpoint.get("dataset_type") != "msasl_npz":
        raise ValueError("Only a validated msasl_npz checkpoint can serve uploaded ASL video")
    gloss_vocab = {index: gloss for index, gloss in enumerate(classes)}
    vocab_size = len(classes)
    model = SignRecognitionTransformer(
        num_keypoints=recognition_model_config["num_keypoints"],
        d_model=recognition_model_config["d_model"],
        nhead=recognition_model_config["nhead"],
        num_encoder_layers=recognition_model_config["num_encoder_layers"],
        vocab_size=vocab_size,
        dropout=recognition_model_config["dropout"],
        use_velocity=recognition_model_config.get("use_velocity", False),
        use_presence=recognition_model_config.get("use_presence", False),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    NUM_FRAMES = int(recognition_model_config["num_frames"])
    calibration_path = RECOGNITION_CHECKPOINT.parent / "calibration.json"
    if not _confidence_override and calibration_path.is_file():
        calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
        selected = calibration.get("selected") if calibration.get("target_met") else None
        if selected is not None:
            RECOGNITION_MIN_CONFIDENCE = float(selected["threshold"])
            RECOGNITION_THRESHOLD_SOURCE = "validation_calibration"
    print(f"Loaded recognition model from {RECOGNITION_CHECKPOINT} with {vocab_size} classes.")
        
except Exception as e:
    print(f"Recognition unavailable: {e}")

@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "project": "ASL-ISL Translation Engine (AITE)",
        "model_loaded": model is not None,
        "recognition_enabled": RECOGNITION_ENABLED,
        "recognition_status": "available" if model is not None else "manual_gloss_input_required",
        "recognition_scope": "MSASL-100 isolated signs" if model is not None else None,
        "recognition_min_confidence": RECOGNITION_MIN_CONFIDENCE if model is not None else None,
        "recognition_threshold_source": RECOGNITION_THRESHOLD_SOURCE if model is not None else None,
        "avatar_mode": "landmark_playback_with_illustrative_fallback",
        "landmark_glosses_available": len(LANDMARK_ASSETS),
        "recorded_sentence_playbacks_available": len(CSLRT_SENTENCE_ASSETS),
    }


def _render_avatar_frame(glosses: list[str], frame_number: int, frame_count: int) -> np.ndarray:
    """Render a lightweight, clearly labelled 2D demo avatar without external assets."""
    canvas = np.full((480, 854, 3), (28, 24, 20), dtype=np.uint8)
    progress = frame_number / max(frame_count - 1, 1)
    active = min(int(progress * len(glosses)), len(glosses) - 1)
    pulse = int(12 * np.sin(progress * np.pi * 4))
    # torso, head, and moving arms
    cv2.ellipse(canvas, (427, 360), (122, 170), 0, 180, 360, (80, 134, 201), -1)
    cv2.circle(canvas, (427, 165), 73, (145, 183, 225), -1)
    shoulder_y = 270
    swing = int(75 * np.sin(progress * np.pi * 6))
    cv2.line(canvas, (355, shoulder_y), (275, 350 + swing), (145, 183, 225), 28)
    cv2.line(canvas, (499, shoulder_y), (579, 350 - swing), (145, 183, 225), 28)
    cv2.circle(canvas, (275, 350 + swing), 20 + pulse // 4, (182, 211, 241), -1)
    cv2.circle(canvas, (579, 350 - swing), 20 + pulse // 4, (182, 211, 241), -1)
    cv2.putText(canvas, "AITE 2D AVATAR DEMO", (35, 50), cv2.FONT_HERSHEY_SIMPLEX, .75, (236, 221, 173), 2, cv2.LINE_AA)
    cv2.putText(canvas, "ISL GLOSS", (35, 412), cv2.FONT_HERSHEY_SIMPLEX, .55, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(canvas, glosses[active][:28], (35, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    return canvas


@app.post("/translate")
async def translate_text(body: dict):
    """Convert manual ASL gloss/text using the local, draft rule baseline."""
    asl_gloss = str(body.get("asl_gloss") or body.get("gloss") or "").strip()
    if not asl_gloss:
        return JSONResponse({"error": "Missing or empty asl_gloss"}, status_code=400)
    if translator is None:
        return JSONResponse(
            {"error": "Translation rules are unavailable; enter ISL gloss manually."},
            status_code=503,
        )
    isl_gloss = translator.translate_gloss_string(asl_gloss)
    if not isl_gloss:
        return JSONResponse({"error": "No translatable gloss tokens were provided."}, status_code=400)
    return {
        "asl_gloss": asl_gloss,
        "isl_gloss": isl_gloss,
        "translation_mode": "draft_rule_based",
        "review_required": True,
    }


@app.post("/generate-avatar")
async def generate_avatar(body: dict):
    """Play a recorded pose sequence when available, otherwise show a labelled demo."""
    isl_gloss = str(body.get("isl_gloss") or body.get("gloss") or "").strip()
    source_sentence = str(body.get("source_sentence") or "").strip()
    glosses = isl_gloss.split()
    if not glosses:
        return JSONResponse({"error": "Missing or empty isl_gloss"}, status_code=400)
    glosses = glosses[:20]
    filename = f"avatar-{uuid.uuid4().hex}.mp4"
    output_path = GENERATED_DIR / filename
    raw_path = GENERATED_DIR / f".{filename}.raw.mp4"
    asset_gloss = next((AVATAR_ALIASES.get(token.lower(), token.lower()) for token in glosses
                        if AVATAR_ALIASES.get(token.lower(), token.lower()) in LANDMARK_ASSETS), None)
    mode = "illustrative_2d"
    source_gloss = None
    source_signer = None
    source_sentence_label = None
    sentence_asset, sentence_match, sentence_match_score = _find_cslrt_sentence_asset(
        source_sentence, allow_approximate=True
    )
    if sentence_asset is None:
        sentence_asset, sentence_match, sentence_match_score = _find_cslrt_sentence_asset(isl_gloss)
    if sentence_asset:
        try:
            # CSLRT consists of short still-frame sequences without source FPS
            # metadata. Seven FPS keeps a 20-frame sentence visible for almost
            # three seconds instead of flashing past at the old 30 FPS default.
            process_file(
                sentence_asset["path"], save_video=str(raw_path), no_show=True,
                override_text=isl_gloss, playback_fps=7,
            )
            mode = "recorded_sentence_landmark_playback"
            source_sentence_label = sentence_asset["sentence"]
            source_signer = sentence_asset["signer"]
        except (OSError, ValueError, KeyError, IndexError) as error:
            raw_path.unlink(missing_ok=True)
            print(f"Sentence landmark avatar fallback for {source_sentence}: {error}")
    elif asset_gloss:
        source_asset = LANDMARK_ASSETS[asset_gloss][0]
        try:
            process_file(source_asset, save_video=str(raw_path), no_show=True, override_text=isl_gloss)
            mode = "landmark_playback"
            source_gloss = asset_gloss
        except (OSError, ValueError, KeyError, IndexError) as error:
            raw_path.unlink(missing_ok=True)
            print(f"Landmark avatar fallback for {asset_gloss}: {error}")
    if mode == "illustrative_2d":
        fps, frame_count = 12, max(24, len(glosses) * 12)
        writer = cv2.VideoWriter(str(raw_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (854, 480))
        if not writer.isOpened():
            return JSONResponse({"error": "Unable to initialise video encoder"}, status_code=500)
        try:
            for frame_number in range(frame_count):
                writer.write(_render_avatar_frame(glosses, frame_number, frame_count))
        finally:
            writer.release()
    try:
        # OpenCV's mp4v output is not consistently playable in browsers.
        # Re-encode to broadly supported H.264 before exposing the file.
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", str(raw_path), "-c:v", "libx264",
                "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output_path),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        raw_path.unlink(missing_ok=True)
        return JSONResponse(
            {"error": "ffmpeg with H.264 support is required to generate browser-playable avatar videos"},
            status_code=500,
        )
    finally:
        raw_path.unlink(missing_ok=True)
    return {
        "video_url": f"/generated/{filename}",
        "mime_type": "video/mp4",
        "mode": mode,
        "source_gloss": source_gloss,
        "source_sentence": source_sentence_label,
        "source_signer": source_signer,
        "sentence_match": sentence_match,
        "sentence_match_score": sentence_match_score,
    }

@app.post("/extract-keypoints")
async def extract_keypoints(file: UploadFile = File(...)):
    return JSONResponse(
        {"error": "Still-image recognition is not supported; upload an isolated-sign video."},
        status_code=410,
    )

@app.post("/predict-gloss")
async def predict_gloss(file: UploadFile = File(...)):
    return JSONResponse(
        {"error": "Still-image recognition is not supported; upload an isolated-sign video."},
        status_code=410,
    )

@app.post("/predict-sequence")
async def predict_sequence(file: UploadFile = File(...)):
    if not RECOGNITION_ENABLED or model is None:
        return JSONResponse(
            {"error": "Recognition is unavailable in this demo. Enter ASL gloss manually."},
            status_code=503,
        )
        
    allowed_types = {"video/mp4", "video/webm", "video/quicktime", "application/octet-stream"}
    if file.content_type and file.content_type not in allowed_types:
        return JSONResponse({"error": "Upload an MP4, WebM, or MOV video."}, status_code=415)
    contents = await file.read(MAX_UPLOAD_BYTES + 1)
    if not contents:
        return JSONResponse({"error": "The uploaded video is empty."}, status_code=400)
    if len(contents) > MAX_UPLOAD_BYTES:
        return JSONResponse({"error": "The uploaded video exceeds the size limit."}, status_code=413)

    suffix = Path(file.filename or "upload.mp4").suffix.lower()
    if suffix not in {".mp4", ".webm", ".mov"}:
        suffix = ".mp4"
    tmp_path = output_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(contents)
            tmp_path = Path(tmp.name)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".npz") as tmp_output:
            output_path = Path(tmp_output.name)
        command = [
            str(PROJECT_DIR / ".venv-cslrt/bin/python"),
            str(PROJECT_DIR / "scripts/extract_msasl_single_video.py"),
            "--input-video", str(tmp_path),
            "--output-file", str(output_path),
            "--num-frames", str(NUM_FRAMES),
        ]
        completed = subprocess.run(
            command,
            cwd=PROJECT_DIR,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=120,
        )
        if completed.returncode != 0:
            return JSONResponse(
                {"error": "The video could not be decoded or no signer was detected."},
                status_code=422,
            )
        with np.load(output_path) as payload:
            stacked = payload["keypoints"].astype(np.float32)
            tracking = json.loads(str(payload["tracking_json"].item()))
    except subprocess.TimeoutExpired:
        return JSONResponse({"error": "Video processing timed out."}, status_code=408)
    except (OSError, ValueError, KeyError):
        return JSONResponse({"error": "The uploaded video could not be processed."}, status_code=422)
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)
        if output_path is not None:
            output_path.unlink(missing_ok=True)

    if stacked.shape != (NUM_FRAMES, 75, 3):
        return JSONResponse({"error": "Unexpected extracted feature shape."}, status_code=500)
    if tracking["pose_coverage"] < 0.80 or tracking["hand_coverage"] < 0.10:
        return JSONResponse(
            {"error": "Signer tracking quality is too low. Keep the upper body and both hands visible."},
            status_code=422,
        )
    kps_tensor = torch.FloatTensor(stacked).unsqueeze(0).to(device)  # (1, T, 75, 3)
    
    with torch.no_grad():
        outputs = model(kps_tensor)
        probs = torch.softmax(outputs, dim=1)[0]
        top_values, top_indices = probs.topk(min(5, len(gloss_vocab)))

    confidence = float(top_values[0].item())
    accepted = confidence >= RECOGNITION_MIN_CONFIDENCE
    predicted_gloss = gloss_vocab.get(top_indices[0].item(), "UNKNOWN")
    return {
        "gloss": predicted_gloss if accepted else "UNKNOWN",
        "candidate_gloss": predicted_gloss,
        "confidence": confidence,
        "accepted": accepted,
        "top5": [
            {"gloss": gloss_vocab[index.item()], "confidence": float(value.item())}
            for value, index in zip(top_values, top_indices)
        ],
        "tracking": tracking,
        "scope": "MSASL-100 isolated signs",
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
