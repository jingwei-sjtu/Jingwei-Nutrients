import argparse
import os

import numpy as np
import pandas as pd
import torch
import xarray as xr
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import build_model
from units import DEFAULT_INPUT_COLUMNS, InferenceDataset, load_json


WOA_VARIABLES = ["o_an", "O_an", "A_an", "I_an", "C_an", "M_an"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", nargs="+", required=True)
    parser.add_argument("--normalization-json", default=None)
    parser.add_argument("--en4-dir", required=True)
    parser.add_argument("--woa-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--start-year", type=int, default=None)
    parser.add_argument("--end-year", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def kelvin_to_celsius(x):
    return x - 273.15


def normalize_longitude(lon):
    return (lon + 180.0) % 360.0 - 180.0


def load_checkpoint(path, fallback_normalization, device):
    checkpoint = torch.load(path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        normalization = checkpoint.get("normalization", fallback_normalization)
        config = checkpoint.get("model_config", {})
    else:
        state_dict = checkpoint
        normalization = fallback_normalization
        config = {}
    if normalization is None:
        raise RuntimeError(f"normalization is missing for {path}")
    input_columns = normalization.get("input_columns", DEFAULT_INPUT_COLUMNS)
    num_features = config.get("num_features", len(input_columns) + 7)
    model = build_model(
        num_features=num_features,
        embed_dim=config.get("embed_dim", 64),
        embed_dim_foundation=config.get("embed_dim_foundation", 64),
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model, normalization, input_columns


def build_dataframe(en4_path, woa_dir):
    name = os.path.basename(en4_path)
    yyyymm = name.split("_")[-1].split(".")[0]
    year = int(yyyymm[:4])
    month = int(yyyymm[4:6])
    woa_path = os.path.join(woa_dir, f"woa_monthly_{month:02d}.nc")
    with xr.open_dataset(en4_path) as ds_en4, xr.open_dataset(woa_path) as ds_woa:
        lon_raw = ds_en4["lon"].values
        lon = normalize_longitude(lon_raw - 0.5)
        lon_order = np.argsort(lon)
        lon = lon[lon_order]
        lat = ds_en4["lat"].values + 0.5
        depth = ds_en4["depth"].values
        temperature = kelvin_to_celsius(ds_en4["temperature"].squeeze().values[:, :, lon_order])
        salinity = ds_en4["salinity"].squeeze().values[:, :, lon_order]
        woa = ds_woa[WOA_VARIABLES].interp(lat=lat, lon=lon, method="nearest")
        arrays = {var: woa[var].values for var in WOA_VARIABLES}
    zz, yy, xx = np.meshgrid(depth, lat, lon, indexing="ij")
    data = {
        "year": np.full(zz.size, year),
        "month": np.full(zz.size, month),
        "depth": zz.ravel(),
        "lat": yy.ravel(),
        "lon": xx.ravel(),
        "Temperature": temperature.ravel(),
        "Salinity": salinity.ravel(),
    }
    for var in WOA_VARIABLES:
        data[var] = arrays[var].ravel()
    return pd.DataFrame(data), year, month, depth, lat, lon


def predict_grid(df, models, normalizations, input_columns, shape, batch_size, num_workers, device):
    sums = {task: np.zeros(df.shape[0], dtype=np.float64) for task in ["N", "P", "Si"]}
    counts = {task: np.zeros(df.shape[0], dtype=np.float64) for task in ["N", "P", "Si"]}
    for model, normalization, cols in zip(models, normalizations, input_columns):
        dataset = InferenceDataset(df, cols, normalization)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        out = {task: [] for task in ["N", "P", "Si"]}
        with torch.no_grad():
            for x in loader:
                x = x.to(device)
                pred = model(x)
                for idx, task in enumerate(["N", "P", "Si"]):
                    stats = normalization["target_stats"][task]
                    values = pred[idx].cpu().numpy().reshape(-1) * stats["std"] + stats["mean"]
                    out[task].append(np.maximum(values, 0.0))
        valid_index = dataset.valid_index.to_numpy()
        for task in ["N", "P", "Si"]:
            if out[task]:
                values = np.concatenate(out[task])
                sums[task][valid_index] += values
                counts[task][valid_index] += 1.0
    grids = {}
    for task, key in [("N", "pred_N"), ("P", "pred_P"), ("Si", "pred_Si")]:
        flat = np.full(df.shape[0], np.nan, dtype=np.float32)
        mask = counts[task] > 0
        flat[mask] = (sums[task][mask] / counts[task][mask]).astype(np.float32)
        grids[key] = flat.reshape(shape)
    return grids


def valid_npz(path):
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return False
    try:
        with np.load(path) as data:
            return {"pred_N", "pred_P", "pred_Si", "lon", "lat", "depth"}.issubset(set(data.files))
    except Exception:
        return False


def save_npz(path, arrays):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp.npz"
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)
    fallback = load_json(args.normalization_json) if args.normalization_json else None
    models = []
    normalizations = []
    input_columns = []
    for path in args.model_path:
        model, normalization, cols = load_checkpoint(path, fallback, device)
        models.append(model)
        normalizations.append(normalization)
        input_columns.append(cols)
    files = sorted(f for f in os.listdir(args.en4_dir) if f.endswith(".nc"))
    for file_name in tqdm(files, desc="inference"):
        yyyymm = file_name.split("_")[-1].split(".")[0]
        year = int(yyyymm[:4])
        if args.start_year is not None and year < args.start_year:
            continue
        if args.end_year is not None and year > args.end_year:
            continue
        month = int(yyyymm[4:6])
        output_path = os.path.join(args.output_dir, f"{year}.{month:02d}.npz")
        if valid_npz(output_path):
            continue
        df, year, month, depth, lat, lon = build_dataframe(os.path.join(args.en4_dir, file_name), args.woa_dir)
        grids = predict_grid(
            df,
            models,
            normalizations,
            input_columns,
            (len(depth), len(lat), len(lon)),
            args.batch_size,
            args.num_workers,
            device,
        )
        grids.update({"lon": lon, "lat": lat, "depth": depth})
        save_npz(output_path, grids)
        print(f"saved {output_path}", flush=True)


if __name__ == "__main__":
    main()
