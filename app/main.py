"""
app/main.py

FastAPI backend exposing:
- GET /health
- POST /predict (multipart file upload OR JSON with base64)
"""

# ---------------------------------------------------------------------------
# TensorFlow / Keras compatibility + clean logging
# ---------------------------------------------------------------------------
# Some trained .h5 models in this project were saved with the legacy
# TF‑Keras stack. Newer Keras 3 strict loading can fail on Lambda layers
# unless we explicitly enable the legacy implementation.
import os
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
# Reduce noisy TensorFlow C++ logs (0=all, 1=INFO, 2=WARNING, 3=ERROR)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
# Suppress Python-level warnings globally (deprecation, etc.)
os.environ.setdefault("PYTHONWARNINGS", "ignore")

import warnings

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
import numpy as np
import io, sys, tempfile
from typing import Optional, List
from pydantic import BaseModel
import logging

# Suppress all warnings at Python level (including tensorflow / tf_keras)
warnings.filterwarnings("ignore")

# Additionally silence TensorFlow's own Python logger so that
# "WARNING:tensorflow:From ..." messages don't spam the console.
try:
    import tensorflow as _tf  # type: ignore

    _tf_logger = _tf.get_logger()
    _tf_logger.setLevel("ERROR")
    logging.getLogger("tensorflow").setLevel(logging.ERROR)
except Exception:
    # If TensorFlow is not available yet, just skip; model import will still work.
    pass

# ---------------------------------------------------------------------------
# Local imports
# NOTE:
# There is a third‑party namespace package called "src" which can shadow this
# project's local `src/` directory when importing `from src import ...`.
# To avoid that conflict, we explicitly add our project's `src` directory to
# `sys.path` and then import the modules directly.
# ---------------------------------------------------------------------------
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import utils  # type: ignore
import model as model_module  # type: ignore

app = FastAPI(title="Model Deployment API", version="1.0")

# Configure basic logging
logger = logging.getLogger("deepguard_api")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)

# Try to load an IMAGE model if present
# Default to the trained hybrid model file name; can be overridden via MODEL_PATH env var.
# Always resolve the path relative to the project root so that cwd doesn't matter.
_raw_model_path = os.environ.get("MODEL_PATH", "hybrid_deepfake_model.h5")
if os.path.isabs(_raw_model_path):
    MODEL_PATH = _raw_model_path
else:
    MODEL_PATH = os.path.join(ROOT_DIR, _raw_model_path)

_loaded_model = None
_model_load_error: Optional[str] = None
try:
    if os.path.exists(MODEL_PATH):
        logger.info(f"Loading IMAGE model from: {MODEL_PATH}")
        _loaded_model = model_module.load_model_from_checkpoint(MODEL_PATH)
        logger.info(f"Image model loaded successfully: name={getattr(_loaded_model, 'name', 'unknown')}")
    else:
        _model_load_error = f"Model file not found at path: {MODEL_PATH}"
        logger.error(_model_load_error)
except Exception as e:
    _loaded_model = None
    _model_load_error = f"Error loading model from '{MODEL_PATH}': {e}"
    logger.exception(_model_load_error)

# Optional VIDEO model + feature extractor
_raw_video_model_path = os.environ.get("VIDEO_MODEL_PATH", "video_deepfake_model.h5")
if os.path.isabs(_raw_video_model_path):
    VIDEO_MODEL_PATH = _raw_video_model_path
else:
    VIDEO_MODEL_PATH = os.path.join(ROOT_DIR, _raw_video_model_path)

_video_model = None
_video_model_error: Optional[str] = None
_video_feature_extractor = None

try:
    if os.path.exists(VIDEO_MODEL_PATH):
        logger.info(f"Loading VIDEO model from: {VIDEO_MODEL_PATH}")
        _video_model = model_module.load_model_from_checkpoint(VIDEO_MODEL_PATH)
        _video_feature_extractor = model_module.build_video_feature_extractor()
        logger.info(f"Video model loaded successfully: name={getattr(_video_model, 'name', 'unknown')}")
    else:
        _video_model_error = f"Video model file not found at path: {VIDEO_MODEL_PATH}"
        logger.warning(_video_model_error)
except Exception as e:
    _video_model = None
    _video_feature_extractor = None
    _video_model_error = f"Error loading video model from '{VIDEO_MODEL_PATH}': {e}"
    logger.exception(_video_model_error)


class PredictResponse(BaseModel):
    # Match the structure returned by model_module.predict_from_input
    prediction: str
    probabilities: List[List[float]]


@app.get("/health")
def health():
    """
    Simple health check.
    Returns whether the model is loaded and exposes basic debug info.
    """
    return {
        "status": "ok",
        "model_loaded": _loaded_model is not None,
        "model_path": MODEL_PATH,
        "model_error": _model_load_error,
        "video_model_loaded": _video_model is not None,
        "video_model_path": VIDEO_MODEL_PATH,
        "video_model_error": _video_model_error,
    }

@app.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...)):
    """
    Accepts an uploaded image file, returns prediction.
    """
    global _loaded_model, _video_model, _video_feature_extractor
    if _loaded_model is None:
        # Do NOT silently fall back to an untrained model.
        # This would give meaningless predictions (often always one class).
        detail = _model_load_error or (
            "Model is not loaded. Ensure MODEL_PATH points to a valid trained model "
            "(.h5 file) and restart the API server."
        )
        raise HTTPException(status_code=500, detail=detail)

    filename = file.filename or "uploaded_file"
    _, ext = os.path.splitext(filename)
    ext = ext.lower()

    video_exts = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv", ".webm"}

    # ----------------------- VIDEO INPUT -----------------------
    if ext in video_exts:
        if _video_model is None or _video_feature_extractor is None:
            detail = _video_model_error or (
                "Video model is not loaded. Ensure VIDEO_MODEL_PATH points to a valid "
                "trained video model (.h5) and that TensorFlow/OpenCV are installed."
            )
            raise HTTPException(status_code=500, detail=detail)

        contents = await file.read()
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp_file:
                tmp_file.write(contents)
                tmp_path = tmp_file.name

            # Use unified helper that supports both images & videos
            result = model_module.predict_from_input_unified(
                _loaded_model,
                tmp_path,
                input_type="video",
                video_model=_video_model,
                feature_extractor=_video_feature_extractor,
            )
            return JSONResponse(content=result)
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Could not process video: {e}")
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass

    # ----------------------- IMAGE INPUT -----------------------
    contents = await file.read()
    try:
        from PIL import Image

        img = Image.open(io.BytesIO(contents)).convert("RGB")
        arr = np.asarray(img.resize((224, 224)))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read image: {e}")

    result = model_module.predict_from_input(_loaded_model, arr)
    return JSONResponse(content=result)

if __name__ == "__main__":
    # Run using the already imported `app` instance instead of a string path
    # to avoid import issues when executed as `python app/main.py`.
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
