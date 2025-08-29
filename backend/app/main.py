from fastapi import FastAPI, UploadFile, File, HTTPException, Form  # type: ignore
from fastapi.middleware.cors import CORSMiddleware  # type: ignore
from pydantic import BaseModel  # type: ignore
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import random
import csv
import io


class PredictRequest(BaseModel):
    symbol: str
    model: str
    start_date: Optional[str] = None
    end_date: Optional[str] = None


class TimeSeriesPoint(BaseModel):
    date: str
    actual: Optional[float] = None
    predicted: Optional[float] = None
    open: Optional[float] = None
    high: Optional[float] = None
    low: Optional[float] = None
    close: Optional[float] = None
    volume: Optional[int] = None


class PredictResponse(BaseModel):
    symbol: str
    model: str
    current_close: float
    next_day_prediction: float
    accuracy_pct: float
    delta_from_previous: float
    trend: str
    confidence: Optional[float]
    time_series: List[TimeSeriesPoint]


class AccuracyResponse(BaseModel):
    accuracies: Dict[str, float]


app = FastAPI(title="Stock Price Analysis & Prediction API", version="0.1.0")

# CORS (allow all origins for local dev)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


SUPPORTED_MODELS = [
    "Linear Regression",
    "Random Forest",
    "SVM",
    "KNN",
    "XGBoost",
    "LightGBM",
]


@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok", "time": datetime.utcnow().isoformat()}


@app.get("/symbols")
async def list_symbols() -> Dict[str, List[str]]:
    # In a real app, pull from DB or config
    return {"symbols": ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN"]}


def _generate_synthetic_series(num_days: int, base_price: float) -> List[TimeSeriesPoint]:
    series: List[TimeSeriesPoint] = []
    current_date = datetime.utcnow() - timedelta(days=num_days)
    price = base_price
    for _ in range(num_days):
        change = random.uniform(-2.0, 2.0)
        open_price = price + random.uniform(-1.0, 1.0)
        high_price = max(open_price, price + abs(change) + random.uniform(0.0, 1.0))
        low_price = min(open_price, price - abs(change) - random.uniform(0.0, 1.0))
        close_price = price + change
        volume = int(random.uniform(1_000_000, 10_000_000))
        series.append(
            TimeSeriesPoint(
                date=current_date.strftime("%Y-%m-%d"),
                open=round(open_price, 2),
                high=round(high_price, 2),
                low=round(low_price, 2),
                close=round(close_price, 2),
                volume=volume,
            )
        )
        price = close_price
        current_date += timedelta(days=1)
    return series


def _inject_predictions(series: List[TimeSeriesPoint]) -> List[TimeSeriesPoint]:
    # Create a simple predicted line: previous close + small noise
    predicted_series: List[TimeSeriesPoint] = []
    prev_close: Optional[float] = None
    for point in series:
        pred = point.close if prev_close is None else prev_close + random.uniform(-1.0, 1.0)
        predicted_series.append(
            TimeSeriesPoint(
                date=point.date,
                actual=point.close,
                predicted=round(pred, 2),
                open=point.open,
                high=point.high,
                low=point.low,
                close=point.close,
                volume=point.volume,
            )
        )
        prev_close = point.close
    return predicted_series


@app.post("/predict", response_model=PredictResponse)
async def predict(payload: PredictRequest) -> PredictResponse:
    if payload.model not in SUPPORTED_MODELS:
        raise HTTPException(status_code=400, detail="Unsupported model")

    # Parse dates if provided
    try:
        if payload.start_date:
            _ = datetime.fromisoformat(payload.start_date)
        if payload.end_date:
            _ = datetime.fromisoformat(payload.end_date)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD")

    series = _generate_synthetic_series(num_days=120, base_price=200.0)
    series = _inject_predictions(series)

    current_close = series[-1].actual if series[-1].actual is not None else series[-1].close or 200.0
    next_prediction = round(current_close + random.uniform(-2.0, 2.0), 2)
    accuracy = round(random.uniform(85.0, 98.0), 2)
    delta = round(current_close - (series[-2].actual or series[-2].close or current_close), 2)
    trend = "up" if next_prediction >= current_close else "down"
    confidence = round(random.uniform(0.6, 0.95), 2)

    return PredictResponse(
        symbol=payload.symbol,
        model=payload.model,
        current_close=round(current_close, 2),
        next_day_prediction=next_prediction,
        accuracy_pct=accuracy,
        delta_from_previous=delta,
        trend=trend,
        confidence=confidence,
        time_series=series,
    )


@app.get("/metrics/accuracies", response_model=AccuracyResponse)
async def model_accuracies() -> AccuracyResponse:
    accuracies: Dict[str, float] = {
        "Linear Regression": 88.5,
        "Random Forest": 92.1,
        "SVM": 90.3,
        "KNN": 87.4,
        "XGBoost": 94.2,
        "LightGBM": 93.7,
    }
    return AccuracyResponse(accuracies=accuracies)


@app.post("/upload", response_model=PredictResponse)
async def upload_csv(
    file: UploadFile = File(...),
    symbol: Optional[str] = Form(None),
    model: Optional[str] = Form("Uploaded CSV"),
) -> PredictResponse:
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported")

    raw = await file.read()
    try:
        text = raw.decode("utf-8")
    except Exception:
        try:
            text = raw.decode("latin-1")
        except Exception:
            raise HTTPException(status_code=400, detail="Unable to decode CSV file")

    reader = csv.DictReader(io.StringIO(text))
    rows = list(reader)
    if not rows:
        raise HTTPException(status_code=400, detail="CSV is empty")

    # Normalize field names
    field_map = {k.lower().strip(): k for k in reader.fieldnames or []}

    def get_val(row: Dict[str, str], key: str) -> Optional[float]:
        if key not in field_map:
            return None
        rawv = row.get(field_map[key])
        if rawv is None or rawv == "":
            return None
        try:
            return float(rawv)
        except Exception:
            return None

    def get_date(row: Dict[str, str]) -> str:
        for k in ("date", "day", "timestamp"):
            if k in field_map:
                return str(row.get(field_map[k]) or "")
        # fallback: generate sequential dates if not present
        return ""

    time_series: List[TimeSeriesPoint] = []
    generated_dates_start = datetime.utcnow() - timedelta(days=len(rows))

    for idx, row in enumerate(rows):
        d = get_date(row)
        if not d:
            d = (generated_dates_start + timedelta(days=idx)).strftime("%Y-%m-%d")
        point = TimeSeriesPoint(
            date=d,
            open=get_val(row, "open"),
            high=get_val(row, "high"),
            low=get_val(row, "low"),
            close=get_val(row, "close") or get_val(row, "actual"),
            volume=int(get_val(row, "volume") or 0),
            actual=get_val(row, "actual") or get_val(row, "close"),
            predicted=get_val(row, "predicted"),
        )
        time_series.append(point)

    # Basic KPIs from CSV
    # Use last two closes/actuals when available
    if len(time_series) < 2:
        raise HTTPException(status_code=400, detail="CSV must contain at least 2 rows")

    last = time_series[-1]
    prev = time_series[-2]

    current_close = (last.actual if last.actual is not None else last.close) or 0.0
    prev_close = (prev.actual if prev.actual is not None else prev.close) or current_close

    # Simple next-day prediction as 5-day SMA if possible
    tail = time_series[-5:]
    sma_vals = [p.actual if p.actual is not None else p.close for p in tail if (p.actual is not None or p.close is not None)]
    if sma_vals:
        next_day_prediction = round(sum(sma_vals) / len(sma_vals), 2)
    else:
        next_day_prediction = round(current_close, 2)

    delta = round(current_close - prev_close, 2)
    trend = "up" if next_day_prediction >= current_close else "down"

    # Confidence/accuracy placeholders (no true labels beyond CSV)
    accuracy = 90.0
    confidence = 0.85

    return PredictResponse(
        symbol=symbol or "CSV",
        model=model or "Uploaded CSV",
        current_close=round(float(current_close), 2),
        next_day_prediction=next_day_prediction,
        accuracy_pct=accuracy,
        delta_from_previous=delta,
        trend=trend,
        confidence=confidence,
        time_series=time_series,
    )
