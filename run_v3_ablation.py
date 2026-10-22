"""Run isolated Sleep-Mamba V3 folds and aggregate complete experiments."""

import argparse
import importlib.util
import os
import subprocess
import sys


VARIANTS = (
    "v3_context",
    "v3_transition",
    "v3_balanced_fusion",
    "v3_fixed_gate",
)


def _run(command, environment):
    print("Running:", " ".join(command), flush=True)
    subprocess.run(command, check=True, env=environment)


def _is_complete(train_dir, fold_index):
    prediction_file = os.path.join(
        train_dir, str(fold_index), "fold_test_predictions.npz")
    return os.path.isfile(prediction_file)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_variant", required=True, choices=VARIANTS)
    parser.add_argument(
        "--config_file", default="config/sleepedfx_v3_200e.py")
    parser.add_argument(
        "--results_dir", default="./results_v3_edf78_10fold_200e")
    parser.add_argument("--from_fold", type=int, default=0)
    parser.add_argument("--to_fold", type=int, default=None)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument(
        "--gpu", type=int, default=None,
        help="Physical GPU index exposed to the V3 subprocess, e.g. 1.")
    parser.add_argument(
        "--skip_completed", action="store_true",
        help="Keep folds that already contain fold_test_predictions.npz.")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    train_script = os.path.join(script_dir, "train_v3.py")
    aggregate_script = os.path.join(script_dir, "aggregate_folds.py")
    config_path = os.path.abspath(args.config_file)

    spec = importlib.util.spec_from_file_location(
        "sleep_mamba_v3_runner_config", config_path)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    n_folds = int(config_module.train["n_folds"])
    to_fold = n_folds - 1 if args.to_fold is None else args.to_fold
    if (args.from_fold < 0 or to_fold < args.from_fold
            or to_fold >= n_folds):
        raise ValueError(
            "Invalid fold range {}..{} for {} folds".format(
                args.from_fold, to_fold, n_folds))

    variant_root = os.path.abspath(os.path.join(
        args.results_dir, args.model_variant))
    train_dir = os.path.join(variant_root, "train")
    log_dir = os.path.join(variant_root, "logs")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    environment = os.environ.copy()
    if args.gpu is not None:
        environment["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        print(
            "Using physical GPU {} (visible as cuda:0 in train_v3.py).".format(
                args.gpu),
            flush=True,
        )

    for fold_index in range(args.from_fold, to_fold + 1):
        if args.skip_completed and _is_complete(train_dir, fold_index):
            print("Skipping completed fold {}.".format(fold_index), flush=True)
            continue
        command = [
            sys.executable,
            train_script,
            "--config_file", config_path,
            "--fold_idx", str(fold_index),
            "--output_dir", train_dir,
            "--log_file", os.path.join(
                log_dir, "fold_{}.log".format(fold_index)),
            "--random_seed", str(args.random_seed),
            "--model_variant", args.model_variant,
            "--restart",
        ]
        _run(command, environment)

    complete_folds = [
        fold_index for fold_index in range(n_folds)
        if _is_complete(train_dir, fold_index)
    ]
    if len(complete_folds) == n_folds:
        if not os.path.isfile(aggregate_script):
            raise FileNotFoundError(
                "aggregate_folds.py must be beside run_v3_ablation.py")
        _run([
            sys.executable,
            aggregate_script,
            "--train_dir", train_dir,
            "--output", os.path.join(variant_root, "cv_results.npz"),
            "--expected_folds", str(n_folds),
        ], environment)
    else:
        missing = sorted(set(range(n_folds)) - set(complete_folds))
        print(
            "Completed folds: {}. Aggregation is skipped; missing: {}".format(
                complete_folds, missing),
            flush=True,
        )


if __name__ == "__main__":
    main()
