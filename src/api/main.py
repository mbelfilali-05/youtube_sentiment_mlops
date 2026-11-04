import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.api.schemas import BatchRequest, BatchResponse, Prediction
from src.utils.text_cleaning import clean_text

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "models" / "sentiment_pipeline.joblib"
LABELS = {-1: "Negative", 0: "Neutral", 1: "Positive"}

(ROOT / "logs").mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler(ROOT / "logs" / "api.log"),
              logging.StreamHandler()],
)
log = logging.getLogger("api")

state = {"model": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # The model is loaded ONCE at startup, not on every request
    try:
        state["model"] = joblib.load(MODEL_PATH)
        log.info("Model loaded from %s", MODEL_PATH)
    except Exception:
        log.exception("Failed to load model")
    yield
    state["model"] = None


app = FastAPI(title="YouTube Sentiment API", version="1.0.0", lifespan=lifespan)

# The Chrome extension calls the API from another origin: CORS must allow it
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    loaded = state["model"] is not None
    return {"status": "ok" if loaded else "degraded",
            "model_loaded": loaded,
            "version": app.version}


@app.post("/predict_batch", response_model=BatchResponse)
def predict_batch(req: BatchRequest):
    model = state["model"]
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    t0 = time.perf_counter()
    try:
        cleaned = [clean_text(c) for c in req.comments]
        probas = model.predict_proba(cleaned)
        classes = model.classes_
    except Exception:
        log.exception("Prediction failed")
        raise HTTPException(status_code=500, detail="Prediction failed")

    predictions = []
    counts = {"Negative": 0, "Neutral": 0, "Positive": 0}
    for original, clean, p in zip(req.comments, cleaned, probas):
        if not clean:
            # Nothing left after cleaning (e.g. only emojis): no signal
            label, conf = 0, 0.0
        else:
            idx = int(p.argmax())
            label, conf = int(classes[idx]), float(p[idx])
        name = LABELS[label]
        counts[name] += 1
        predictions.append(Prediction(text=original, label=label,
                                      sentiment=name,
                                      confidence=round(conf, 4)))

    total = len(predictions)
    percentages = {k: round(100 * v / total, 1) for k, v in counts.items()}
    elapsed = (time.perf_counter() - t0) * 1000
    log.info("predict_batch n=%d time=%.1fms", total, elapsed)
    return BatchResponse(predictions=predictions, total=total, counts=counts,
                         percentages=percentages,
                         inference_ms=round(elapsed, 2))