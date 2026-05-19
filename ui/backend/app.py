import io
import json
import os
import sys
from pathlib import Path
from typing import Optional
from contextlib import asynccontextmanager

import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from torchvision import transforms

# Adjust path to import from src
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.mhsdf import MHSDF
from src.utils.text_vectorizer import TextVectorizer
from src.data.transforms import get_base_transforms

# Global models
EASY_OCR = None
STAGE1_MODEL = None
STAGE1_VECTORIZER = None
STAGE2_MODEL = None
STAGE2_VECTORIZER = None

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRANSFORM = get_base_transforms(224)

# Stage 2 classes: 0: Racist, 1: Sexist, 2: Homophobe, 3: Religion, 4: OtherHate
STAGE2_CLASSES = {
    0: "Racist",
    1: "Sexist",
    2: "Homophobe",
    3: "Religion",
    4: "OtherHate"
}

def load_stage_model(ckpt_path: Path, num_classes: int):
    if not ckpt_path.exists():
        print(f"WARNING: Checkpoint {ckpt_path} not found.")
        return None, None
    print(f"Loading checkpoint from {ckpt_path} onto {DEVICE}...")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    vocab = ckpt["vocab"]
    
    vectorizer = TextVectorizer(min_freq=1)
    vectorizer.vocab = vocab
    
    model = MHSDF(vocab_size=len(vocab), num_classes=num_classes, multilabel=False, freeze_cnn=True)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.to(DEVICE)
    model.eval()
    return model, vectorizer

def load_all_models():
    global EASY_OCR, STAGE1_MODEL, STAGE1_VECTORIZER, STAGE2_MODEL, STAGE2_VECTORIZER
    
    # Load PyTorch Models
    s1_path = PROJECT_ROOT / "checkpoints" / "p7_stage1_caps" / "best.pt"
    STAGE1_MODEL, STAGE1_VECTORIZER = load_stage_model(s1_path, 2)
    
    s2_path = PROJECT_ROOT / "checkpoints" / "p7_stage2_caps" / "best.pt"
    STAGE2_MODEL, STAGE2_VECTORIZER = load_stage_model(s2_path, 5)
    
    # Initialize OCR
    try:
        import easyocr
        print("Loading EasyOCR...")
        EASY_OCR = easyocr.Reader(['en'], gpu=torch.cuda.is_available())
    except Exception as e:
        print(f"EasyOCR failed to load: {e}")
        
    # VLM intentionally not used for this backend

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_all_models()
    yield

app = FastAPI(title="Meme Vibe Classifier API", lifespan=lifespan)

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def run_model(model, vectorizer, img_tensor, text: str):
    text = text.strip() if text else ""
    ids_batch, lengths = vectorizer.encode([text], max_len=64)
    if lengths[0] == 0:
        lengths[0] = 1
        ids_batch[0][0] = vectorizer.vocab.get("<PAD>", 0)

    text_ids = torch.tensor(ids_batch, dtype=torch.long).to(DEVICE)
    lengths_t = torch.tensor(lengths, dtype=torch.long).to(DEVICE)
    img_t = img_tensor.to(DEVICE)

    with torch.no_grad():
        logits = model(img_t, text_ids, lengths_t)
        probs = F.softmax(logits, dim=-1)[0]
    return probs.cpu().numpy()

@app.post("/predict")
async def predict(
    image: UploadFile = File(...),
    text: Optional[str] = Form("")
):
    try:
        image_bytes = await image.read()
        pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img_tensor = TRANSFORM(pil_img).unsqueeze(0)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {e}")

    ocr_text = ""
    models_used = ["ResNet18", "BiLSTM"]

    # Run OCR if text is empty
    if not text or not text.strip():
        if EASY_OCR:
            import numpy as np
            results = EASY_OCR.readtext(np.array(pil_img), detail=0)
            print(f"DEBUG OCR Results: {results}")
            ocr_text = " ".join(results)
            models_used.append("EasyOCR")
            
        final_text = ocr_text.strip()
    else:
        final_text = text.strip()

    if STAGE1_MODEL is None or STAGE2_MODEL is None:
        return {"error": "Models not loaded"}

    # Stage 1: Binary (NotHate vs Hate)
    s1_probs = run_model(STAGE1_MODEL, STAGE1_VECTORIZER, img_tensor, final_text)
    is_hate = bool(s1_probs[1] > s1_probs[0])
    s1_conf = float(max(s1_probs))
    
    binary_result = "Hateful" if is_hate else "Not Hateful"
    binary_probs = {"NotHate": float(s1_probs[0]), "Hateful": float(s1_probs[1])}

    primary_class = "NotHate"
    multiclass_conf = s1_conf
    s2_probs_dict = {}

    # Stage 2: Fine-grained Hate categories (if Hate)
    if is_hate:
        s2_probs = run_model(STAGE2_MODEL, STAGE2_VECTORIZER, img_tensor, final_text)
        s2_pred_idx = int(s2_probs.argmax())
        primary_class = STAGE2_CLASSES[s2_pred_idx]
        multiclass_conf = float(s2_probs[s2_pred_idx])
        s2_probs_dict = {STAGE2_CLASSES[i]: float(s2_probs[i]) for i in range(5)}

    return {
        "primary_class": primary_class,
        "confidence": multiclass_conf,
        "binary_result": binary_result,
        "binary_probs": binary_probs,
        "multiclass_probs": s2_probs_dict,
        "ocr_text": ocr_text,
        "final_text_used": final_text,
        "models_used": models_used
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
