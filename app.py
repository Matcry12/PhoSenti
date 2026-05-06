from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ── Load model once at startup ──────────────────────────────────────────────
CHECKPOINT = "checkpoints/best"
MAX_LENGTH = 256
LABELS = ["negative", "neutral", "positive"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)
model = AutoModelForSequenceClassification.from_pretrained(CHECKPOINT)
model.to(device)
model.eval()

# ── App ──────────────────────────────────────────────────────────────────────
app = FastAPI(title="PhoSenti")


class PredictRequest(BaseModel):
    text: str


@app.get("/health")
def health():
    return {"status": "ok", "device": str(device)}


@app.post("/predict")
def predict(req: PredictRequest):
    inputs = tokenizer(
        req.text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = F.softmax(logits, dim=-1).squeeze().tolist()
    predicted_idx = int(torch.argmax(logits))
    return {
        "text": req.text,
        "sentiment": LABELS[predicted_idx],
        "confidence": round(probs[predicted_idx], 4),
        "scores": {LABELS[i]: round(probs[i], 4) for i in range(3)},
    }


@app.get("/", response_class=HTMLResponse)
def ui():
    return """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>PhoSenti</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }

    body {
      font-family: 'Segoe UI', system-ui, sans-serif;
      background: #f5f5f5;
      min-height: 100vh;
      display: flex;
      align-items: center;
      justify-content: center;
      padding: 24px;
    }

    .card {
      background: white;
      border-radius: 16px;
      box-shadow: 0 4px 24px rgba(0,0,0,0.08);
      padding: 40px;
      width: 100%;
      max-width: 560px;
    }

    h1 {
      font-size: 1.6rem;
      font-weight: 700;
      color: #1a1a1a;
      margin-bottom: 4px;
    }

    .subtitle {
      font-size: 0.9rem;
      color: #888;
      margin-bottom: 28px;
    }

    textarea {
      width: 100%;
      height: 120px;
      border: 1.5px solid #e0e0e0;
      border-radius: 10px;
      padding: 14px;
      font-size: 1rem;
      font-family: inherit;
      resize: vertical;
      transition: border-color 0.2s;
      outline: none;
    }

    textarea:focus { border-color: #4f6ef7; }

    button {
      margin-top: 14px;
      width: 100%;
      padding: 13px;
      background: #4f6ef7;
      color: white;
      border: none;
      border-radius: 10px;
      font-size: 1rem;
      font-weight: 600;
      cursor: pointer;
      transition: background 0.2s;
    }

    button:hover { background: #3a57e8; }
    button:disabled { background: #a0aec0; cursor: not-allowed; }

    .result { margin-top: 28px; }

    .sentiment-badge {
      display: inline-block;
      padding: 6px 18px;
      border-radius: 20px;
      font-size: 1rem;
      font-weight: 700;
      text-transform: capitalize;
      margin-bottom: 20px;
    }

    .positive  { background: #dcfce7; color: #16a34a; }
    .negative  { background: #fee2e2; color: #dc2626; }
    .neutral   { background: #f1f5f9; color: #475569; }

    .bars { display: flex; flex-direction: column; gap: 10px; }

    .bar-row { display: flex; align-items: center; gap: 10px; }

    .bar-label {
      width: 72px;
      font-size: 0.85rem;
      color: #555;
      text-transform: capitalize;
    }

    .bar-track {
      flex: 1;
      background: #f0f0f0;
      border-radius: 6px;
      height: 10px;
      overflow: hidden;
    }

    .bar-fill {
      height: 100%;
      border-radius: 6px;
      transition: width 0.5s ease;
    }

    .fill-positive { background: #22c55e; }
    .fill-negative { background: #ef4444; }
    .fill-neutral  { background: #94a3b8; }

    .bar-pct {
      width: 42px;
      font-size: 0.82rem;
      color: #888;
      text-align: right;
    }

    .error { color: #dc2626; margin-top: 16px; font-size: 0.9rem; }

    #result { display: none; }
  </style>
</head>
<body>
  <div class="card">
    <h1>PhoSenti</h1>
    <p class="subtitle">Vietnamese sentiment analysis · PhoBERT fine-tuned on UIT-VSFC</p>

    <textarea id="input" placeholder="Nhập câu tiếng Việt... (e.g. Thầy giảng rất dễ hiểu)"></textarea>
    <button id="btn" onclick="analyze()">Analyze</button>

    <div id="result" class="result">
      <span id="badge" class="sentiment-badge"></span>
      <div class="bars" id="bars"></div>
    </div>

    <p id="error" class="error"></p>
  </div>

  <script>
    async function analyze() {
      const text = document.getElementById("input").value.trim();
      if (!text) return;

      const btn = document.getElementById("btn");
      btn.disabled = true;
      btn.textContent = "Analyzing…";
      document.getElementById("error").textContent = "";
      document.getElementById("result").style.display = "none";

      try {
        const res = await fetch("/predict", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ text }),
        });

        if (!res.ok) throw new Error("Server error");
        const data = await res.json();

        // Badge
        const badge = document.getElementById("badge");
        badge.textContent = data.sentiment + "  " + (data.confidence * 100).toFixed(1) + "%";
        badge.className = "sentiment-badge " + data.sentiment;

        // Bars
        const bars = document.getElementById("bars");
        bars.innerHTML = "";
        for (const [label, score] of Object.entries(data.scores)) {
          bars.innerHTML += `
            <div class="bar-row">
              <span class="bar-label">${label}</span>
              <div class="bar-track">
                <div class="bar-fill fill-${label}" style="width: ${(score * 100).toFixed(1)}%"></div>
              </div>
              <span class="bar-pct">${(score * 100).toFixed(1)}%</span>
            </div>`;
        }

        document.getElementById("result").style.display = "block";
      } catch (e) {
        document.getElementById("error").textContent = "Something went wrong. Is the server running?";
      } finally {
        btn.disabled = false;
        btn.textContent = "Analyze";
      }
    }

    document.getElementById("input").addEventListener("keydown", e => {
      if (e.key === "Enter" && e.ctrlKey) analyze();
    });
  </script>
</body>
</html>
"""


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
