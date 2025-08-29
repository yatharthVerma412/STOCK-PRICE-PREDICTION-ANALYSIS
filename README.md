# Frontend - Stock Price Analysis & Next-Day Prediction

An interactive dashboard built with React (Vite + TypeScript), Tailwind CSS, Recharts, and Lightweight Charts.

## Prerequisites
- Node.js 18+ (22 recommended)
- Backend running on http://localhost:8000

## Setup
```
cd frontend
npm install
```

## Run (dev)
```
npm run dev
```
- Open `http://localhost:5173`
- Vite dev proxy routes `http://localhost:5173/api` to `http://localhost:8000`

## Environment (optional)
- To override the API base, create `.env.local` with:
```
VITE_API_URL=http://localhost:8000
```
If not set, the app uses `/api` which is proxied to `localhost:8000` in dev.

## Build
```
npm run build
npm run preview
```

## Features
- Header: Title, subtitle, current date, stock selector, theme switch
- Inputs: Date range picker, optional CSV upload
- Model selection: Linear Regression, Random Forest, SVM, KNN, XGBoost, LightGBM
- KPIs: Current price, next-day prediction, accuracy %, delta with up/down indicator
- Charts: Actual vs Predicted, Candlestick OHLC, Moving Averages (7/30), Volume, Accuracies
- Extras: Search, CSV export, dark/light theme, footer disclaimer

## Notes
- Data comes from the FastAPI backend endpoints.
