# PhoSenti — Vietnamese Sentiment Analysis

Fine-tuned [PhoBERT](https://github.com/VinAIResearch/PhoBERT) for 3-class sentiment classification on Vietnamese student feedback.

## Results

| Metric | Score |
|---|---|
| Accuracy | 93.3% |
| F1 Macro | 0.84 |
| F1 Negative | 0.95 |
| F1 Neutral | 0.57 |
| F1 Positive | 0.95 |

Evaluated on the [UIT-VSFC](https://huggingface.co/datasets/ura-hcmut/UIT-VSFC) test set (3,166 examples).

## Setup

```bash
pip install transformers datasets torch scikit-learn fastapi uvicorn accelerate
```

## Usage

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch, torch.nn.functional as F

model = AutoModelForSequenceClassification.from_pretrained("checkpoints/best")
tokenizer = AutoTokenizer.from_pretrained("checkpoints/best")
model.eval()

def predict(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = F.softmax(logits, dim=-1).squeeze().tolist()
    labels = ["negative", "neutral", "positive"]
    idx = int(torch.argmax(logits))
    return {"sentiment": labels[idx], "confidence": round(probs[idx], 4)}

print(predict("Thầy giảng rất dễ hiểu và nhiệt tình hỗ trợ sinh viên."))
# {"sentiment": "positive", "confidence": 0.97}
```

## Training

See `notebook.ipynb` for the full pipeline:
- Dataset: UIT-VSFC (11,426 train / 1,584 val / 3,166 test)
- Base model: `vinai/phobert-base`
- Fine-tuning: 3 epochs, AdamW, lr=2e-5
- Best checkpoint selected by validation F1 macro

## Stack

- PyTorch + HuggingFace Transformers
- PhoBERT (Vietnamese BERT)
- UIT-VSFC dataset
