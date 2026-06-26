import json
import math
import os
import random

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import Dataset


DEFAULT_INPUT_COLUMNS = [
    "Temperature",
    "Salinity",
    "o_an",
    "O_an",
    "A_an",
    "I_an",
    "C_an",
    "M_an",
]

TARGET_COLUMNS = {
    "N": "NITRATE",
    "P": "PHO",
    "Si": "SIL",
}

TARGET_LIMITS = {
    "N": (0.0, 60.0),
    "P": (0.0, 20.0),
    "Si": (0.0, 300.0),
}

BASE_FEATURE_COLUMNS = [
    "year_norm",
    "month_sin",
    "month_cos",
    "x_coord",
    "y_coord",
    "z_coord",
    "depth_norm",
]

TASK_ORDER = ["N", "P", "Si"]


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_training_dataframe(paths, station_flag_column="Station_Flag", drop_station_flag=True):
    frames = []
    for path in paths:
        frame = pd.read_csv(path)
        if drop_station_flag and station_flag_column in frame.columns:
            frame = frame[frame[station_flag_column] != 1].copy()
        frames.append(frame)
    columns = list(dict.fromkeys(column for frame in frames for column in frame.columns))
    frames = [frame.reindex(columns=columns) for frame in frames]
    return pd.concat(frames, ignore_index=True, copy=False)


def assign_fold(year):
    year = int(year)
    ranges = [(1965, 1974), (1975, 1984), (1985, 1994), (1995, 2004), (2005, 2014), (2015, 2023)]
    for idx, (start, end) in enumerate(ranges):
        if start <= year <= end:
            return idx
    return -1


def fold_name(fold):
    names = {
        0: "1965-1974",
        1: "1975-1984",
        2: "1985-1994",
        3: "1995-2004",
        4: "2005-2014",
        5: "2015-2023",
    }
    return names[int(fold)]


def standardize_columns(df):
    df = df.copy()
    aliases = {
        "YEAR": "year",
        "MON": "month",
        "LAT": "lat",
        "LON": "lon",
        "DEPTH_LAYER": "depth",
    }
    for source, target in aliases.items():
        if source in df.columns and target not in df.columns:
            df[target] = df[source]
    return df


def add_features(df, year_min=1960.0, year_max=2024.0):
    df = standardize_columns(df)
    df = df.copy()
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12.0)
    df["year_norm"] = (df["year"] - year_min + 1.0) / (year_max - year_min + 1.0)
    lat_rad = np.radians(df["lat"])
    lon_rad = np.radians(df["lon"])
    df["x_coord"] = np.cos(lat_rad) * np.cos(lon_rad)
    df["y_coord"] = np.cos(lat_rad) * np.sin(lon_rad)
    df["z_coord"] = np.sin(lat_rad)
    return df


def fit_feature_stats(df, input_cols, depth_column="depth"):
    df = standardize_columns(df)
    stats = {}
    for col in input_cols:
        values = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        mean = float(values.mean())
        std = float(values.std(ddof=0))
        stats[col] = {"mean": mean, "std": std if np.isfinite(std) and std > 0 else 1.0}
    depth = pd.to_numeric(df[depth_column], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    depth_max = float(depth.max()) if len(depth) else 1.0
    return {"input": stats, "depth_max": depth_max if depth_max > 0 else 1.0}


def filter_target_rows(df, task, input_cols):
    target = TARGET_COLUMNS[task]
    lower, upper = TARGET_LIMITS[task]
    cols = input_cols + [target]
    out = df.dropna(subset=cols).copy()
    out = out[(out[target] > lower) & (out[target] < upper)].copy()
    return out


def fit_target_stats(df, input_cols):
    stats = {}
    for task, target in TARGET_COLUMNS.items():
        values = filter_target_rows(df, task, input_cols)[target].astype(float)
        mean = float(values.mean())
        std = float(values.std(ddof=0))
        stats[task] = {"column": target, "mean": mean, "std": std if np.isfinite(std) and std > 0 else 1.0}
    return stats


def build_normalization(df_train, input_cols):
    feature_stats = fit_feature_stats(df_train, input_cols)
    target_stats = fit_target_stats(df_train, input_cols)
    return {
        "input_columns": list(input_cols),
        "base_feature_columns": list(BASE_FEATURE_COLUMNS),
        "feature_stats": feature_stats["input"],
        "target_stats": target_stats,
        "depth_max": feature_stats["depth_max"],
        "year_min": 1960.0,
        "year_max": 2024.0,
    }


def apply_normalization(df, input_cols, normalization):
    df = add_features(df, normalization.get("year_min", 1960.0), normalization.get("year_max", 2024.0))
    for col in input_cols:
        mean = normalization["feature_stats"][col]["mean"]
        std = normalization["feature_stats"][col]["std"]
        df[col] = (df[col] - mean) / std
    depth_max = normalization["depth_max"] if normalization["depth_max"] > 0 else 1.0
    df["depth_norm"] = df["depth"] / depth_max
    return df


class NutrientTaskDataset(Dataset):
    def __init__(self, df, task, input_cols, normalization):
        df = filter_target_rows(standardize_columns(df), task, input_cols)
        df = apply_normalization(df, input_cols, normalization)
        target = TARGET_COLUMNS[task]
        target_stats = normalization["target_stats"][task]
        feature_cols = BASE_FEATURE_COLUMNS + list(input_cols)
        x = df[feature_cols].astype(np.float32).to_numpy()
        y = df[[target]].astype(np.float32).to_numpy()
        y = (y - target_stats["mean"]) / target_stats["std"]
        mask = ~np.isnan(x).any(axis=1) & ~np.isnan(y).any(axis=1)
        self.x = x[mask]
        self.y = y[mask].astype(np.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return torch.from_numpy(self.x[idx]), torch.from_numpy(self.y[idx])


class InferenceDataset(Dataset):
    def __init__(self, df, input_cols, normalization):
        df = apply_normalization(standardize_columns(df), input_cols, normalization)
        feature_cols = BASE_FEATURE_COLUMNS + list(input_cols)
        x = df[feature_cols].astype(np.float32).to_numpy()
        self.valid_mask = ~np.isnan(x).any(axis=1)
        self.valid_index = df.index[self.valid_mask]
        self.x = x[self.valid_mask]

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return torch.from_numpy(self.x[idx])


def pcgrad_step(model, optimizer, losses):
    optimizer.zero_grad()
    task_grads = []
    params = list(model.parameters())
    for loss in losses:
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        task_grads.append([None if p.grad is None else p.grad.detach().clone() for p in params])
    optimizer.zero_grad()
    order = list(range(len(losses)))
    random.shuffle(order)
    final_grads = [torch.zeros_like(p) for p in params]
    for i in order:
        grad_i = task_grads[i]
        for j in order:
            if i == j:
                continue
            grad_j = task_grads[j]
            dot = torch.tensor(0.0, device=params[0].device)
            norm = torch.tensor(0.0, device=params[0].device)
            for gi, gj in zip(grad_i, grad_j):
                if gi is not None and gj is not None:
                    dot = dot + torch.sum(gi * gj)
                    norm = norm + torch.sum(gj * gj)
            if dot < 0:
                coeff = dot / (norm + 1e-8)
                grad_i = [None if gi is None else gi - coeff * gj if gj is not None else gi for gi, gj in zip(grad_i, grad_j)]
        for k, gi in enumerate(grad_i):
            if gi is not None:
                final_grads[k] = final_grads[k] + gi
    for param, grad in zip(params, final_grads):
        param.grad = grad
    optimizer.step()


def train_epoch(model, loaders, optimizer, loss_fn, device):
    model.train()
    iterators = {task: iter(loader) for task, loader in loaders.items()}
    steps = max(len(loader) for loader in loaders.values())
    sums = {task: 0.0 for task in TASK_ORDER}
    for _ in range(steps):
        losses = []
        raw_losses = {}
        for idx, task in enumerate(TASK_ORDER):
            try:
                x, y = next(iterators[task])
            except StopIteration:
                iterators[task] = iter(loaders[task])
                x, y = next(iterators[task])
            x = x.to(device)
            y = y.to(device)
            output = model(x)[idx]
            raw = loss_fn(output, y)
            sigma = getattr(model, f"log_sigma_{task}")
            losses.append(torch.exp(-sigma) * raw + sigma)
            raw_losses[task] = raw.item()
        pcgrad_step(model, optimizer, losses)
        for task in TASK_ORDER:
            sums[task] += raw_losses[task]
    return {task: sums[task] / steps for task in TASK_ORDER}


def validate_epoch(model, loader, loss_fn, task, device):
    model.eval()
    idx = TASK_ORDER.index(task)
    total = 0.0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            total += loss_fn(model(x)[idx], y).item()
    return total / max(len(loader), 1)


def predict_task(model, loader, task, normalization, device):
    model.eval()
    idx = TASK_ORDER.index(task)
    stats = normalization["target_stats"][task]
    preds = []
    trues = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            pred = model(x)[idx].cpu().numpy() * stats["std"] + stats["mean"]
            true = y.numpy() * stats["std"] + stats["mean"]
            preds.append(pred.reshape(-1))
            trues.append(true.reshape(-1))
    if not preds:
        return np.array([]), np.array([])
    return np.concatenate(preds), np.concatenate(trues)


def regression_metrics(y_true, y_pred):
    if len(y_true) == 0:
        return {"R2": np.nan, "RMSE": np.nan, "MAE": np.nan}
    r2 = r2_score(y_true, y_pred) if len(y_true) > 1 else np.nan
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    return {"R2": float(r2), "RMSE": float(rmse), "MAE": float(mae)}


def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
