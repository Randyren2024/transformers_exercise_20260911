import os
from pathlib import Path

from flask import Flask, jsonify, render_template, request

from model import TinyGPTService

BASE_DIR = Path(__file__).resolve().parent
CHECKPOINT = Path(os.getenv("TINY_GPT_CHECKPOINT", BASE_DIR / "model" / "tiny_gpt_step54_best.pt"))
TOKENIZER = Path(os.getenv("TINY_GPT_TOKENIZER", BASE_DIR / "model" / "step43_bpe_8000.json"))
DEVICE = os.getenv("TINY_GPT_DEVICE", "")

app = Flask(__name__)
service = TinyGPTService(CHECKPOINT, TOKENIZER, DEVICE or None)

@app.get("/")
def index():
    return render_template("index.html")

@app.get("/api/health")
def health():
    return jsonify({"status": "ok", "device": str(service.device)})

@app.post("/api/chat")
def chat():
    payload = request.get_json(silent=True) or {}
    message = str(payload.get("message", "")).strip()
    if not message:
        return jsonify({"error": "message is required"}), 400
    if len(message) > 1000:
        return jsonify({"error": "message is too long"}), 400
    try:
        max_new = max(1, min(int(payload.get("max_new_tokens", 64)), 128))
    except (TypeError, ValueError):
        max_new = 64
    response = service.generate(message, max_new)
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)