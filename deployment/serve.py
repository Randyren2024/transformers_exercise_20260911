import os
from pathlib import Path

os.environ["TINY_GPT_CHECKPOINT"] = str(Path(__file__).resolve().parent / "models" / "tiny_gpt_v2_step_8000.pt")
os.environ["TINY_GPT_TOKENIZER"] = str(Path(__file__).resolve().parent / "models" / "step43_bpe_8000.json")

from waitress import serve
from app import app

print("Starting Waitress on http://127.0.0.1:5000")
serve(app, host="127.0.0.1", port=5000)
