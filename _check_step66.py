import os
d = "/content/drive/MyDrive/transformers_exercise_20260911/artifacts/step66"
if os.path.exists(d):
    for name in sorted(os.listdir(d)):
        f = os.path.join(d, name)
        sz = os.path.getsize(f)
        print(f"{name}: {sz} bytes")
else:
    print("Directory does not exist yet")
