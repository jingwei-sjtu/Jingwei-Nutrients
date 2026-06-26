import argparse
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from model import build_model
from units import (
    DEFAULT_INPUT_COLUMNS,
    TASK_ORDER,
    NutrientTaskDataset,
    assign_fold,
    build_normalization,
    fold_name,
    load_training_dataframe,
    predict_task,
    regression_metrics,
    save_json,
    seed_everything,
    train_epoch,
    validate_epoch,
)


class EarlyStopping:
    def __init__(self, patience=20, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_score = -np.inf
        self.counter = 0
        self.stop = False

    def update(self, score):
        if score > self.best_score + self.min_delta:
            self.best_score = score
            self.counter = 0
            return True
        self.counter += 1
        self.stop = self.counter >= self.patience
        return False


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-csv", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4, 5])
    parser.add_argument("--input-cols", nargs="+", default=DEFAULT_INPUT_COLUMNS)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--embed-dim", type=int, default=64)
    parser.add_argument("--embed-dim-foundation", type=int, default=64)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def build_loaders(df_train, df_eval, input_cols, normalization, batch_size, num_workers):
    loaders = {}
    eval_loaders = {}
    for task in TASK_ORDER:
        train_dataset = NutrientTaskDataset(df_train, task, input_cols, normalization)
        eval_dataset = NutrientTaskDataset(df_eval, task, input_cols, normalization)
        if len(train_dataset) == 0:
            raise RuntimeError(f"{task} training dataset is empty")
        if len(eval_dataset) == 0:
            raise RuntimeError(f"{task} validation dataset is empty")
        loaders[task] = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            drop_last=False,
        )
        eval_loaders[task] = DataLoader(
            eval_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            drop_last=False,
        )
    return loaders, eval_loaders


def run_fold(df_all, fold, args, device):
    name = fold_name(fold)
    fold_dir = os.path.join(args.output_dir, f"fold_{fold}_{name.replace('-', '_')}")
    os.makedirs(fold_dir, exist_ok=True)
    df_eval = df_all[df_all["fold"] == fold].copy()
    df_train = df_all[df_all["fold"] != fold].copy()
    if df_train.empty or df_eval.empty:
        raise RuntimeError(f"fold {fold} has empty train or validation data")

    normalization = build_normalization(df_train, args.input_cols)
    save_json(normalization, os.path.join(fold_dir, "normalization.json"))
    train_loaders, eval_loaders = build_loaders(
        df_train,
        df_eval,
        args.input_cols,
        normalization,
        args.batch_size,
        args.num_workers,
    )

    model = build_model(
        num_features=len(args.input_cols) + 7,
        embed_dim=args.embed_dim,
        embed_dim_foundation=args.embed_dim_foundation,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    loss_fn = torch.nn.MSELoss()
    early_stopping = EarlyStopping(patience=args.patience)
    log_rows = []
    best_path = os.path.join(fold_dir, "best_model.pt")

    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loaders, optimizer, loss_fn, device)
        val_loss = {task: validate_epoch(model, eval_loaders[task], loss_fn, task, device) for task in TASK_ORDER}
        fold_metrics = {}
        for task in TASK_ORDER:
            pred, true = predict_task(model, eval_loaders[task], task, normalization, device)
            fold_metrics[task] = regression_metrics(true, pred)
        score = float(np.nanmean([fold_metrics[task]["R2"] for task in TASK_ORDER]))
        row = {
            "epoch": epoch + 1,
            "score": score,
            **{f"train_loss_{task}": train_loss[task] for task in TASK_ORDER},
            **{f"val_loss_{task}": val_loss[task] for task in TASK_ORDER},
            **{f"R2_{task}": fold_metrics[task]["R2"] for task in TASK_ORDER},
            **{f"RMSE_{task}": fold_metrics[task]["RMSE"] for task in TASK_ORDER},
            **{f"MAE_{task}": fold_metrics[task]["MAE"] for task in TASK_ORDER},
        }
        log_rows.append(row)
        print(
            f"fold {fold} epoch {epoch + 1} score {score:.4f} "
            f"R2 N/P/Si {row['R2_N']:.4f}/{row['R2_P']:.4f}/{row['R2_Si']:.4f}",
            flush=True,
        )
        if early_stopping.update(score):
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "normalization": normalization,
                    "input_columns": args.input_cols,
                    "model_config": {
                        "num_features": len(args.input_cols) + 7,
                        "embed_dim": args.embed_dim,
                        "embed_dim_foundation": args.embed_dim_foundation,
                    },
                    "fold": fold,
                    "fold_name": name,
                },
                best_path,
            )
        if early_stopping.stop:
            break

    pd.DataFrame(log_rows).to_csv(os.path.join(fold_dir, "training_log.csv"), index=False)
    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    results = []
    for task in TASK_ORDER:
        pred, true = predict_task(model, eval_loaders[task], task, normalization, device)
        pd.DataFrame({"Observed": true, "Predicted": pred}).to_csv(
            os.path.join(fold_dir, f"test_results_{task}.csv"),
            index=False,
        )
        metrics = regression_metrics(true, pred)
        metrics.update({"fold": fold, "fold_name": name, "task": task})
        results.append(metrics)
    return results


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    seed_everything(args.seed)
    device = torch.device(args.device)
    df_all = load_training_dataframe(args.train_csv)
    if "YEAR" not in df_all.columns:
        raise KeyError("training data must contain YEAR")
    df_all["fold"] = pd.to_numeric(df_all["YEAR"], errors="coerce").apply(assign_fold)
    df_all = df_all[df_all["fold"] != -1].copy()
    all_results = []
    for fold in args.folds:
        all_results.extend(run_fold(df_all, fold, args, device))
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    summary = pd.DataFrame(all_results)
    summary.to_csv(os.path.join(args.output_dir, "kfold_summary.csv"), index=False)
    summary.groupby("task")[["R2", "RMSE", "MAE"]].mean().to_csv(
        os.path.join(args.output_dir, "kfold_average_metrics.csv")
    )


if __name__ == "__main__":
    main()
