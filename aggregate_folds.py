"""Aggregate strictly out-of-fold predictions produced by train.py."""

import argparse
import glob
import os

import numpy as np
from sklearn import metrics


CLASSES = ["W", "N1", "N2", "N3", "REM"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", default="./output/train")
    parser.add_argument("--output", default="./output/cv_results.npz")
    parser.add_argument("--expected_folds", type=int, default=None)
    args = parser.parse_args()

    pattern = os.path.join(args.train_dir, "*", "fold_test_predictions.npz")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError("No fold_test_predictions.npz files found: " + pattern)
    if args.expected_folds is not None and len(files) != args.expected_folds:
        raise RuntimeError(
            "Expected {} folds but found {} under {}".format(
                args.expected_folds, len(files), args.train_dir))

    y_true, y_pred, fold_rows = [], [], []
    variants, parameter_counts = set(), set()
    for path in files:
        with np.load(path) as data:
            true = np.asarray(data["y_true"])
            pred = np.asarray(data["y_pred"])
            fold_idx = int(data["fold_idx"])
            if "model_variant" in data:
                variants.add(str(data["model_variant"]))
            if "n_parameters" in data:
                parameter_counts.add(int(data["n_parameters"]))
        y_true.append(true)
        y_pred.append(pred)
        fold_rows.append((
            fold_idx,
            metrics.accuracy_score(true, pred),
            metrics.f1_score(true, pred, average="macro", zero_division=0),
            metrics.cohen_kappa_score(true, pred),
        ))

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    cm = metrics.confusion_matrix(y_true, y_pred, labels=np.arange(5))
    per_class_f1 = metrics.f1_score(
        y_true, y_pred, labels=np.arange(5), average=None, zero_division=0)
    accuracy = metrics.accuracy_score(y_true, y_pred)
    macro_f1 = metrics.f1_score(y_true, y_pred, average="macro", zero_division=0)
    kappa = metrics.cohen_kappa_score(y_true, y_pred)

    fold_rows = np.asarray(sorted(fold_rows), dtype=np.float64)
    print("fold  accuracy  macro_f1  kappa")
    for row in fold_rows:
        print(f"{int(row[0]):>4}  {row[1]:>8.4f}  {row[2]:>8.4f}  {row[3]:>6.4f}")
    print("\nOverall out-of-fold results")
    if variants:
        print("model_variant=", ",".join(sorted(variants)))
    if parameter_counts:
        print("parameters=", ",".join(map(str, sorted(parameter_counts))))
    print(f"n={len(y_true)}, accuracy={accuracy:.4f}, macro_f1={macro_f1:.4f}, kappa={kappa:.4f}")
    print("Per-class F1:", dict(zip(CLASSES, np.round(per_class_f1, 4))))
    print("Confusion matrix:\n", cm)

    output_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(output_dir, exist_ok=True)
    np.savez(
        args.output,
        y_true=y_true,
        y_pred=y_pred,
        confusion_matrix=cm,
        per_class_f1=per_class_f1,
        accuracy=accuracy,
        macro_f1=macro_f1,
        kappa=kappa,
        fold_metrics=fold_rows,
        class_names=np.asarray(CLASSES),
        model_variant=np.asarray(sorted(variants)),
        n_parameters=np.asarray(sorted(parameter_counts), dtype=np.int64),
    )
    print("Saved:", args.output)


if __name__ == "__main__":
    main()
