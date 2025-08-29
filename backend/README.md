# Backend (FastAPI)

## Setup

1. Create virtual environment
```
python -m venv .venv
```

2. Install dependencies
```
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\python -m pip install -r backend\requirements.txt
```

3. Run server
```
.venv\Scripts\uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000/docs` for API docs.
