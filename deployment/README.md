# TinyGPT Step 54 — Flask deployment

This directory is the first deployment shell for the frozen Step 54 checkpoint.

## Model files

Place these files in deployment/model/:

- tiny_gpt_step54_best.pt
- step43_bpe_8000.json

The binary checkpoint is intentionally not committed to Git.

## Local CPU run

Create a virtual environment and install requirements.txt.

Linux/WSL:

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py

Windows PowerShell:

py -3 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python app.py

Open http://127.0.0.1:5000/

## API

GET /api/health

POST /api/chat

Example JSON body:

{"message":"What is a robot?"}

## Important

The model is loaded once at application startup and reused for requests.
Do not reload the checkpoint inside the request handler.

CPU is the recommended first deployment target. GPU is optional.

For public deployment, do not use Flask's development server directly. Use a production WSGI server such as Waitress on Windows or Gunicorn on Linux. A reverse proxy such as Nginx can sit in front of the WSGI server.

Flask documentation:
https://flask.palletsprojects.com/en/stable/deploying/

PyTorch documents state_dict loading as the normal inference pattern and recommends eval() for inference.