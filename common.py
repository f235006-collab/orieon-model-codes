"""Shared data preparation, evaluation, and output utilities for the ORIEON paper."""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Portable project paths. Place the two supplied CSV files inside data/.
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
GOLD_PATH = DATA_DIR / "gold_clean.csv"
NEWS_PATH = DATA_DIR / "news_with_scores.csv"
RESULTS.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)

# Fixed before model estimation: explicit gold and macro-financial relevance terms.
RELEVANCE_TERMS = [
    "gold", "xau", "bullion", "precious metal", "federal reserve", "fed ",
    "interest rate", "inflation", "consumer price", "cpi", "central bank",
    "us dollar", "u.s. dollar", "dollar index", "treasury yield", "bond yield",
    "monetary policy", "recession", "economic growth", "nonfarm", "payroll",
    "unemployment", "geopolitical", "safe haven", "safe-haven",
]


def metric_dict(y_true, y_pred, current_close):
    """Calculate exactly the metrics reported in the paper."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    current_close = np.asarray(current_close, dtype=float)
    actual_direction = y_true > current_close
    predicted_direction = y_pred > current_close
    nonzero = np.abs(y_true) > 1e-12
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
        "MAPE_pct": float(
            np.mean(np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero])) * 100
        ),
        "Directional_Accuracy_pct": float(
            np.mean(actual_direction == predicted_direction) * 100
        ),
        "n": int(len(y_true)),
    }


def prepare_data():
    """Load, filter, aggregate, and chronologically partition the supplied data."""
    for path in (GOLD_PATH, NEWS_PATH):
        if not path.exists():
            raise FileNotFoundError(
                f"Required file not found: {path}\n"
                "Place gold_clean.csv and news_with_scores.csv in the data/ folder."
            )

    gold = pd.read_csv(GOLD_PATH, parse_dates=["Date"]).sort_values("Date").reset_index(drop=True)
    news = pd.read_csv(NEWS_PATH, parse_dates=["Date"]).sort_values("Date").reset_index(drop=True)

    clean = news["Clean"].fillna("").str.lower()
    pattern = "|".join(re.escape(term) for term in RELEVANCE_TERMS)
    relevant = news[clean.str.contains(pattern, regex=True, na=False)].copy()
    relevant["positive"] = (relevant["Score"] > 0).astype(float)
    relevant["negative"] = (relevant["Score"] < 0).astype(float)

    daily = relevant.groupby("Date").agg(
        Sentiment_Mean=("Score", "mean"),
        Sentiment_Std=("Score", "std"),
        Sentiment_Min=("Score", "min"),
        Sentiment_Max=("Score", "max"),
        Positive_Ratio=("positive", "mean"),
        Negative_Ratio=("negative", "mean"),
        News_Count=("Score", "size"),
    )

    data = gold.join(daily, on="Date")
    sentiment_cols = [
        "Sentiment_Mean", "Sentiment_Std", "Sentiment_Min", "Sentiment_Max",
        "Positive_Ratio", "Negative_Ratio", "News_Count",
    ]
    data["News_Available"] = data["News_Count"].notna().astype(float)
    data[sentiment_cols] = data[sentiment_cols].fillna(0.0)
    data["News_Count_Log"] = np.log1p(data["News_Count"])
    data = data.drop(columns=["News_Count"])

    # y_logret is excluded because it is the next-day target return.
    # y_close_t1 is the prediction target and is never an input feature.
    feature_cols = [
        c for c in data.columns if c not in ["Date", "y_logret", "y_close_t1"]
    ]
    if data[feature_cols + ["y_close_t1"]].isna().any().any():
        raise ValueError("Unexpected missing values remain after preprocessing.")

    n = len(data)
    n_train = int(np.floor(0.70 * n))
    n_val = int(np.floor(0.15 * n))
    split = {
        "n": n,
        "n_train": n_train,
        "n_val": n_val,
        "n_test": n - n_train - n_val,
        "train_start": str(data.Date.iloc[0].date()),
        "train_end": str(data.Date.iloc[n_train - 1].date()),
        "val_start": str(data.Date.iloc[n_train].date()),
        "val_end": str(data.Date.iloc[n_train + n_val - 1].date()),
        "test_start": str(data.Date.iloc[n_train + n_val].date()),
        "test_end": str(data.Date.iloc[-1].date()),
        "relevant_headlines": int(len(relevant)),
        "relevant_share_pct": float(100 * len(relevant) / len(news)),
        "relevant_news_dates": int(relevant.Date.nunique()),
        "feature_count": len(feature_cols),
        "features": feature_cols,
    }

    data.to_csv(ROOT / "processed_data.csv", index=False)
    relevant.to_csv(ROOT / "filtered_news.csv", index=False)
    with open(RESULTS / "data_protocol.json", "w", encoding="utf-8") as f:
        json.dump(split, f, indent=2)
    return data, feature_cols, split


def save_predictions(model, split_name, dates, y_true, y_pred, current_close):
    frame = pd.DataFrame({
        "Date": dates,
        "actual": y_true,
        "predicted": y_pred,
        "current_close": current_close,
    })
    frame.to_csv(RESULTS / f"{model}_{split_name}_predictions.csv", index=False)


def update_metrics(model, split_name, metrics):
    path = RESULTS / "metrics.csv"
    row = {"Model": model, "Split": split_name, **metrics}
    if path.exists():
        table = pd.read_csv(path)
        table = table[~((table.Model == model) & (table.Split == split_name))]
        table = pd.concat([table, pd.DataFrame([row])], ignore_index=True)
    else:
        table = pd.DataFrame([row])
    table.to_csv(path, index=False)
