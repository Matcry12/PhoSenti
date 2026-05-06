# PhoSenti — Project Plan

Vietnamese sentiment analysis API using PhoBERT, fine-tuned on student/course reviews.

---

## Goal

Build a deployable NLP API that classifies Vietnamese text as **negative**, **neutral**, or **positive**, returning confidence scores for all 3 classes.

---

## Input / Output

**Input:**
```json
{ "text": "Thầy giảng rất dễ hiểu và nhiệt tình hỗ trợ sinh viên." }
```

**Output:**
```json
{
  "text": "Thầy giảng rất dễ hiểu và nhiệt tình hỗ trợ sinh viên.",
  "sentiment": "positive",
  "confidence": 0.94,
  "scores": {
    "negative": 0.02,
    "neutral": 0.04,
    "positive": 0.94
  }
}
```

---

## Stack

| Component | Choice |
|---|---|
| Base model | `vinai/phobert-base` (HuggingFace) |
| Fine-tuning | PyTorch + HuggingFace `Trainer` |
| API | FastAPI |
| Tokenizer | `AutoTokenizer` from PhoBERT |
| Dataset | UIT-VSFC (Vietnamese student feedback corpus) |

---

## Phases

### Phase 1 — Data & Baseline
- [ ] Download UIT-VSFC dataset (train/dev/test splits)
- [ ] Explore label distribution and sample texts
- [ ] Write `dataset.py`: tokenize and return `DataLoader`-ready tensors
- [ ] Confirm PhoBERT loads and tokenizes correctly

### Phase 2 — Fine-tuning
- [ ] Write `model.py`: `PhoBertForSequenceClassification` wrapper (3-class head)
- [ ] Write `train.py`: training loop with HuggingFace `Trainer`
- [ ] Train on UIT-VSFC, track loss and accuracy per epoch
- [ ] Evaluate on test set, report F1 per class
- [ ] Save best checkpoint to `checkpoints/`

### Phase 3 — Inference & API
- [ ] Write `predict.py`: load checkpoint, run single-text inference, return all 3 scores
- [ ] Write `app.py`: FastAPI app with `/predict` endpoint
- [ ] Test endpoint manually with `curl` or Postman
- [ ] Add `/health` endpoint

### Phase 4 — Polish
- [ ] Write `README.md` with setup instructions and example curl command
- [ ] Add `requirements.txt`
- [ ] Record a short demo (optional, good for portfolio)

---

## Project Structure (target)

```
PhoSenti/
├── data/
│   └── vsfc/           # raw UIT-VSFC files
├── checkpoints/        # saved model weights
├── dataset.py          # tokenization and DataLoader
├── model.py            # model class
├── train.py            # training script
├── predict.py          # inference utility
├── app.py              # FastAPI app
├── requirements.txt
├── PLAN.md
└── README.md
```

---

## Success Criteria

- F1 ≥ 0.85 on UIT-VSFC test set
- API returns correct JSON shape with all 3 confidence scores
- Cold-start inference under 2 seconds on CPU
