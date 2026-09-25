import json
from pathlib import Path
d = Path("/content/drive/MyDrive/transformers_exercise_20260911/artifacts/step66")
with open(d / "step66_metadata.json") as f:
    meta = json.load(f)
print(json.dumps(meta, indent=2, ensure_ascii=False))
