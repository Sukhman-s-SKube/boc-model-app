import logging
import os
import pickle
import threading
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb
from clickhouse_connect import get_client

from app.server.utils.s3_util import MODELS_BUCKET_DEFAULT, get_s3

log = logging.getLogger("model_service")


CLASS_LABELS = ["cut", "hold", "raise"]
DEFAULT_RATE_MOVE_TOL = float(os.getenv("RATE_MOVE_TOL", "0.0125"))
CH_HOST = os.getenv("CH_HOST", "localhost")
CH_PORT = int(os.getenv("CH_PORT", "9000"))
CH_USER = os.getenv("CH_USER") or None
CH_PASS = os.getenv("CH_PASS") or None
CH_DB = os.getenv("CH_DB", "default")
CH_TABLE = os.getenv("CH_TABLE", "macro_daily")
CH_SECURE = os.getenv("CH_SECURE", "false").strip().lower() in {"1", "true", "yes"}
FEATURE_COLUMNS = [
    name.strip()
    for name in os.getenv("FEATURE_COLUMNS", "rate,cpi,y2,y5,y10,spread_2_10,oil,unemploy").split(",")
    if name.strip()
]


def _get_clickhouse_client():
    return get_client(
        host=CH_HOST,
        port=CH_PORT,
        username=CH_USER,
        password=CH_PASS,
        database=CH_DB,
        secure=CH_SECURE,
    )


def _fetch_macro_dataframe(start: Optional[date], end: Optional[date], horizon: int) -> pd.DataFrame:
    client = _get_clickhouse_client()
    try:
        params: Dict[str, str] = {}
        conds: List[str] = []
        if start:
            conds.append("toDate(`date`) >= %(start)s")
            params["start"] = start.isoformat()
        fetch_end = end + timedelta(days=horizon) if end else None
        if fetch_end:
            conds.append("toDate(`date`) <= %(end)s")
            params["end"] = fetch_end.isoformat()
        column_clause = ", ".join(f"`{c}`" for c in FEATURE_COLUMNS)
        where_clause = f" WHERE {' AND '.join(conds)}" if conds else ""
        query = f"SELECT toDate(`date`) AS d, {column_clause} FROM {CH_TABLE}{where_clause} ORDER BY d ASC"
        df = client.query_df(query, parameters=params)
    finally:
        client.close()
    if df.empty:
        return df
    df = df.rename(columns={"d": "date"})
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    df = df.sort_values("date").drop_duplicates("date")
    df = df.set_index("date")
    full_range = pd.date_range(df.index.min(), df.index.max(), freq="D")
    df = df.reindex(full_range)
    df = df.sort_index()
    df = df.ffill().bfill()
    df = df.astype(np.float64)
    if df.isna().any().any():
        missing_cols = df.columns[df.isna().any()].tolist()
        log.warning(
            "NaNs remain in ClickHouse dataset after forward/back fill for columns %s. Filling with zeros.",
            missing_cols,
        )
        df = df.fillna(0.0)
    df.index.name = "date"
    return df


def _create_windows(
    df: pd.DataFrame,
    seq_len: int,
    horizon: int,
    start_bound: Optional[date],
    end_bound: Optional[date],
    require_future: bool,
) -> Tuple[np.ndarray, Optional[np.ndarray], List[date]]:
    if df.empty:
        return np.zeros((0, seq_len, len(FEATURE_COLUMNS)), dtype=np.float32), None, []
    arr = df[FEATURE_COLUMNS].to_numpy(dtype=np.float32, copy=True)
    dates = df.index.to_pydatetime()
    n = arr.shape[0]
    max_end = n - (horizon if require_future else 0)
    windows: List[np.ndarray] = []
    labels: List[float] = []
    window_dates: List[date] = []
    for end_idx in range(seq_len - 1, max_end):
        window_date = dates[end_idx].date()
        if start_bound and window_date < start_bound:
            continue
        if end_bound and window_date > end_bound:
            continue
        window = arr[end_idx - seq_len + 1 : end_idx + 1, :]
        if not np.isfinite(window).all():
            continue
        if require_future:
            future_idx = end_idx + horizon
            if future_idx >= n:
                break
            future_rate = arr[future_idx, 0]
            if not np.isfinite(future_rate):
                continue
            labels.append(float(future_rate))
        windows.append(window)
        window_dates.append(window_date)
    if not windows:
        empty_X = np.zeros((0, seq_len, arr.shape[1]), dtype=np.float32)
        if require_future:
            return empty_X, np.zeros((0,), dtype=np.float32), []
        return empty_X, None, []
    X = np.stack(windows, axis=0)
    if require_future:
        Y = np.array(labels, dtype=np.float32)
        return X, Y, window_dates
    return np.stack(windows, axis=0), None, window_dates


def _normalize_base_feature_names(feature_count: int, prefer: Optional[List[str]] = None) -> List[str]:
    if prefer and len(prefer) == feature_count:
        return list(prefer)
    if prefer and len(prefer) != feature_count:
        log.warning(
            "BASE_FEATURE_NAMES has %d entries but dataset has %d features; using generic names.",
            len(prefer),
            feature_count,
        )
    return [f"f{i:02d}" for i in range(feature_count)]


def _build_feature_matrix(X: np.ndarray, base_names: List[str]) -> Tuple[np.ndarray, List[str]]:
    if X.ndim != 3:
        raise ValueError(f"Expected X with ndim=3, got shape {getattr(X, 'shape', None)}")
    B, T, F = X.shape
    names = _normalize_base_feature_names(F, base_names)
    arr = X.astype(np.float64, copy=False)
    idx = np.arange(T, dtype=np.float64)
    centered = idx - np.mean(idx)
    denom = float(np.sum(centered**2)) or 1.0
    last = arr[:, -1, :]
    first = arr[:, 0, :]
    prev = arr[:, -2, :] if T > 1 else last
    mean_full = np.mean(arr, axis=1)
    std_full = np.std(arr, axis=1)
    min_full = np.min(arr, axis=1)
    max_full = np.max(arr, axis=1)
    slope_full = np.tensordot(arr, centered, axes=([1], [0])) / denom
    rng_full = max_full - min_full
    window7 = arr[:, -min(7, T) :, :]
    window30 = arr[:, -min(30, T) :, :]
    mean7 = np.mean(window7, axis=1)
    std7 = np.std(window7, axis=1)
    mean30 = np.mean(window30, axis=1)
    std30 = np.std(window30, axis=1)
    rng7 = np.max(window7, axis=1) - np.min(window7, axis=1)
    rng30 = np.max(window30, axis=1) - np.min(window30, axis=1)
    eps = 1e-6
    features: List[np.ndarray] = []
    feature_names: List[str] = []

    def push(name: str, value: np.ndarray) -> None:
        features.append(value.reshape(B, 1))
        feature_names.append(name)

    for i, base in enumerate(names):
        push(f"{base}_last", last[:, i])
        push(f"{base}_mean90", mean_full[:, i])
        push(f"{base}_std90", std_full[:, i])
        push(f"{base}_min90", min_full[:, i])
        push(f"{base}_max90", max_full[:, i])
        push(f"{base}_range90", rng_full[:, i])
        push(f"{base}_slope", slope_full[:, i])
        push(f"{base}_delta_last_first", last[:, i] - first[:, i])
        push(f"{base}_delta_last_prev", last[:, i] - prev[:, i])
        push(f"{base}_zscore_last", (last[:, i] - mean_full[:, i]) / (std_full[:, i] + eps))
        push(f"{base}_mean7", mean7[:, i])
        push(f"{base}_std7", std7[:, i])
        push(f"{base}_mean30", mean30[:, i])
        push(f"{base}_std30", std30[:, i])
        push(f"{base}_range7", rng7[:, i])
        push(f"{base}_range30", rng30[:, i])
        push(f"{base}_mom7", last[:, i] - mean7[:, i])
        push(f"{base}_mom30", last[:, i] - mean30[:, i])
        push(f"{base}_trend_ratio", slope_full[:, i] / (np.abs(mean_full[:, i]) + eps))
        push(f"{base}_vol_ratio", std7[:, i] / (std30[:, i] + eps))
    if names:
        rate_last = last[:, 0]
        rate_mean = mean_full[:, 0]
        rate_slope = slope_full[:, 0]
        for i, base in enumerate(names[1:], start=1):
            push(f"rate_minus_{base}_last", rate_last - last[:, i])
            push(f"rate_minus_{base}_mean", rate_mean - mean_full[:, i])
            push(f"rate_slope_minus_{base}", rate_slope - slope_full[:, i])

    mat = np.concatenate(features, axis=1).astype(np.float32, copy=False)
    mat = np.nan_to_num(mat, nan=0.0, posinf=0.0, neginf=0.0)
    return mat, feature_names


def _predict_with_booster(booster: xgb.Booster, features: np.ndarray, best_iteration: Optional[int]) -> np.ndarray:
    dmat = xgb.DMatrix(features)
    if best_iteration is not None and best_iteration >= 0:
        ntree_limit = int(best_iteration) + 1
        proba = booster.predict(dmat, ntree_limit=ntree_limit)
    else:
        proba = booster.predict(dmat)
    if proba.ndim == 1:
        proba = proba.reshape(-1, len(CLASS_LABELS))
    return proba


def _latest_available_window_date() -> date:
    client = _get_clickhouse_client()
    try:
        df = client.query_df(f"SELECT toDate(max(`date`)) AS d FROM {CH_TABLE}")
    finally:
        client.close()
    if df.empty or df.iloc[0]["d"] is None:
        raise RuntimeError("ClickHouse table returned no rows; cannot determine latest date.")
    return pd.to_datetime(df.iloc[0]["d"]).date()


def load_model_from_s3(model_bucket: str, model_key: str) -> Tuple[xgb.Booster, dict]:
    s3 = get_s3()
    obj = s3.get_object(Bucket=model_bucket, Key=model_key)
    payload = pickle.loads(obj["Body"].read())
    booster_bytes = payload.get("booster")
    if booster_bytes is None:
        raise RuntimeError("Model payload missing booster bytes.")
    booster = xgb.Booster()
    booster.load_model(bytearray(booster_bytes))
    return booster, payload.get("meta", {})


def latest_model_key(model_bucket: str, model_prefix: str) -> Optional[str]:
    s3 = get_s3()
    prefix = f"{model_prefix.rstrip('/')}/"
    cont = None
    keys: List[str] = []
    while True:
        kwargs = {"Bucket": model_bucket, "Prefix": prefix, "MaxKeys": 1000}
        if cont:
            kwargs["ContinuationToken"] = cont
        resp = s3.list_objects_v2(**kwargs)
        for obj in resp.get("Contents", []):
            if obj["Key"].endswith(".pkl"):
                keys.append(obj["Key"])
        if resp.get("IsTruncated"):
            cont = resp.get("NextContinuationToken")
        else:
            break
    if not keys:
        return None
    keys.sort()
    return keys[-1]


class ModelService:
    def __init__(self, model_bucket: Optional[str] = None, model_prefix: str = "boc_policy_classifier"):
        self._bucket = model_bucket or MODELS_BUCKET_DEFAULT
        self._prefix = model_prefix
        self._booster: Optional[xgb.Booster] = None
        self._meta: Dict[str, Any] = {}
        self._model_key: Optional[str] = None
        self._last_loaded: Optional[datetime] = None
        self._lock = threading.Lock()

    def ensure_model_loaded(self) -> None:
        if self._booster is None:
            self.reload_latest_model()

    def reload_latest_model(self) -> Dict[str, Any]:
        with self._lock:
            key = latest_model_key(self._bucket, self._prefix)
            if not key:
                raise RuntimeError("No model found in models bucket/prefix.")
            booster, meta = load_model_from_s3(self._bucket, key)
            self._booster = booster
            self._meta = meta or {}
            self._model_key = key
            self._last_loaded = datetime.now(timezone.utc)
            log.info("Loaded model %s from bucket %s", key, self._bucket)
            return {"model_key": key, "loaded_at": self._last_loaded.isoformat()}

    def predict(self, date_str: Optional[str] = None) -> Dict[str, Any]:
        self.ensure_model_loaded()
        if self._booster is None or not self._meta:
            raise RuntimeError("Model metadata missing; reload the model.")
        expected_seq_len = int(self._meta.get("seq_len", 90))
        horizon = int(self._meta.get("horizon", 30))
        base_names = self._meta.get("base_feature_names") or FEATURE_COLUMNS
        feature_stats = self._meta.get("feature_stats", {})
        clip_min = np.array(feature_stats.get("clip_min") or [], dtype=np.float32)
        clip_max = np.array(feature_stats.get("clip_max") or [], dtype=np.float32)
        feature_mean = np.array(feature_stats.get("mean") or [], dtype=np.float32)
        feature_std = np.array(feature_stats.get("std") or [], dtype=np.float32)
        best_iteration = self._meta.get("best_iteration")
        if best_iteration is not None:
            best_iteration = int(best_iteration)

        if date_str:
            target_date = date.fromisoformat(date_str)
        else:
            target_date = _latest_available_window_date()
        fetch_buffer_days = max(expected_seq_len + horizon + 30, 150)
        fetch_start = target_date - timedelta(days=fetch_buffer_days)
        df = _fetch_macro_dataframe(fetch_start, target_date, 0)
        if df.empty:
            raise RuntimeError("No data returned from ClickHouse for inference window.")
        X_win, _, window_dates = _create_windows(df, expected_seq_len, horizon, target_date, target_date, require_future=False)
        if X_win.shape[0] == 0:
            raise RuntimeError(f"Insufficient history to build a {expected_seq_len}-day window ending {target_date}.")

        feats, _ = _build_feature_matrix(X_win, base_names)
        if clip_min.size == feats.shape[1]:
            feats = np.maximum(feats, clip_min)
        if clip_max.size == feats.shape[1]:
            feats = np.minimum(feats, clip_max)
        drift_score = None
        if feature_mean.size == feats.shape[1] and feature_std.size == feats.shape[1]:
            denom = np.where(feature_std == 0.0, 1.0, feature_std)
            drift_score = np.mean(np.abs(feats - feature_mean) / denom, axis=1)
        proba = _predict_with_booster(self._booster, feats, best_iteration)
        pred_idx = np.argmax(proba, axis=1)
        pred_labels = [CLASS_LABELS[int(i)] for i in pred_idx]
        last_rate = X_win[:, -1, 0].astype(np.float32, copy=False)

        sample = []
        for i in range(min(10, proba.shape[0])):
            sample.append(
                {
                    "label": pred_labels[i],
                    "probabilities": {CLASS_LABELS[j]: float(proba[i, j]) for j in range(proba.shape[1])},
                    "drift_score": None if drift_score is None else float(drift_score[i]),
                    "last_rate": float(last_rate[i]),
                    "window_date": window_dates[i].isoformat() if i < len(window_dates) else target_date.isoformat(),
                }
            )
        pred_counts: Dict[str, int] = {}
        unique, counts = np.unique(pred_idx, return_counts=True)
        for idx, cnt in zip(unique, counts):
            pred_counts[CLASS_LABELS[int(idx)]] = int(cnt)

        result = {
            "status": "ok",
            "date": target_date.isoformat(),
            "n_scored": int(proba.shape[0]),
            "model_bucket": self._bucket,
            "model_key": self._model_key,
            "class_labels": CLASS_LABELS,
            "pred_sample_first10": sample,
            "predicted_distribution": pred_counts,
            "seq_len": int(expected_seq_len),
            "label_tolerance": float(self._meta.get("label_tolerance", DEFAULT_RATE_MOVE_TOL)),
        }
        if drift_score is not None and drift_score.size > 0:
            result["drift_score_mean"] = float(np.mean(drift_score))
            result["drift_score_max"] = float(np.max(drift_score))
        return result


_model_service = ModelService()


def get_model_service() -> ModelService:
    return _model_service
