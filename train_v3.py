"""Train boundary-aware Sleep-Mamba V3 without touching V1/V2 files."""

import argparse
import glob
import importlib.util
import os
import shutil

import numpy as np
import sklearn.metrics as skmetrics
import torch

from data import get_subject_files, load_data
from logger import get_logger
from minibatching_v3 import iterate_context_minibatches
from model_v3 import ModelV3
from network_v3 import MODEL_VARIANTS
from utils import load_seq_ids, print_n_samples_each_class


def _load_experiment_config(config_file):
    spec = importlib.util.spec_from_file_location(
        "sleep_mamba_v3_config", config_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.train.copy()


def _load_split(config, subject_ids):
    subject_files = glob.glob(os.path.join(config["data_dir"], "*.npz"))
    selected_files = []
    for subject_id in subject_ids:
        files = get_subject_files(
            dataset=config["dataset"],
            files=subject_files,
            sid=int(subject_id),
        )
        if not files:
            raise RuntimeError(
                "No NPZ files found for subject {}".format(subject_id))
        selected_files.extend(files)
    return load_data(np.asarray(selected_files))[:2]


def _shift_within_epoch_zero(signal, offset):
    shifted = np.roll(signal, offset, axis=1)
    if offset > 0:
        shifted[:, :offset, ...] = 0
    elif offset < 0:
        shifted[:, offset:, ...] = 0
    return shifted


def _shift_within_epoch_reflect(signal, offset):
    if offset == 0:
        return signal.copy()
    amount = abs(offset)
    pad_width = [(0, 0)] * signal.ndim
    if offset > 0:
        pad_width[1] = (amount, 0)
        padded = np.pad(signal, pad_width, mode="reflect")
        return padded[:, :signal.shape[1], ...]
    pad_width[1] = (0, amount)
    padded = np.pad(signal, pad_width, mode="reflect")
    return padded[:, amount:amount + signal.shape[1], ...]


def _augment_training_data(train_x, train_y, config):
    augmented_x = [np.copy(x) for x in train_x]
    augmented_y = [np.copy(y) for y in train_y]
    if not bool(config.get("augment_signal_full", False)):
        return augmented_x, augmented_y

    percent = float(config.get("signal_shift_percent", 0.1))
    mode = config.get("signal_shift_mode", "within_epoch_zero")
    for index in range(len(augmented_x)):
        offset = int(np.random.uniform(-percent, percent)
                     * augmented_x[index].shape[1])
        if mode == "official_continuous":
            shifted = np.roll(augmented_x[index], offset)
            if offset < 0:
                augmented_x[index] = shifted[:-1]
                augmented_y[index] = augmented_y[index][:-1]
            elif offset > 0:
                augmented_x[index] = shifted[1:]
                augmented_y[index] = augmented_y[index][1:]
            else:
                augmented_x[index] = shifted
        elif mode == "within_epoch_reflect":
            augmented_x[index] = _shift_within_epoch_reflect(
                augmented_x[index], offset)
        elif mode == "within_epoch_zero":
            augmented_x[index] = _shift_within_epoch_zero(
                augmented_x[index], offset)
        else:
            raise ValueError("Unknown signal_shift_mode: {}".format(mode))
        if len(augmented_x[index]) != len(augmented_y[index]):
            raise RuntimeError("augmentation changed signal/label alignment")
    return augmented_x, augmented_y


def _make_minibatches(
        x, y, config, shuffle_idx=None, augment_seq=False):
    return iterate_context_minibatches(
        inputs=x,
        targets=y,
        batch_size=int(config["batch_size"]),
        context_length=int(config["seq_length"]),
        target_length=int(config["target_length"]),
        shuffle_idx=shuffle_idx,
        augment_seq=augment_seq,
    )


def train(
        config_file, fold_idx, output_dir, log_file,
        model_variant="v3_context", restart=False, random_seed=42):
    config = _load_experiment_config(config_file)
    config["model_variant"] = model_variant
    if model_variant not in MODEL_VARIANTS:
        raise ValueError("Invalid V3 model variant")

    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)

    fold_output_dir = os.path.join(output_dir, str(fold_idx))
    if restart and os.path.exists(fold_output_dir):
        shutil.rmtree(fold_output_dir)
    os.makedirs(fold_output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)
    logger = get_logger(log_file, level="info")

    subject_ids = load_seq_ids("{}.txt".format(config["dataset"]))
    fold_subjects = np.array_split(subject_ids, int(config["n_folds"]))
    test_sids = np.asarray(fold_subjects[fold_idx])
    remaining_sids = np.setdiff1d(subject_ids, test_sids)
    n_valid = round(len(remaining_sids) * 0.10)
    np.random.seed(random_seed)
    valid_sids = np.random.choice(
        remaining_sids, size=n_valid, replace=False)
    train_sids = np.setdiff1d(remaining_sids, valid_sids)

    logger.info("Model variant: {}".format(model_variant))
    logger.info("Context length: {}, target length: {}".format(
        config["seq_length"], config["target_length"]))
    logger.info("Train SIDs: ({}) {}".format(len(train_sids), train_sids))
    logger.info("Valid SIDs: ({}) {}".format(len(valid_sids), valid_sids))
    logger.info("Test SIDs: ({}) {}".format(len(test_sids), test_sids))

    train_x, train_y = _load_split(config, train_sids)
    valid_x, valid_y = _load_split(config, valid_sids)
    test_x, test_y = _load_split(config, test_sids)
    logger.info("Training set: nights={}, epochs={}".format(
        len(train_y), sum(len(y) for y in train_y)))
    print_n_samples_each_class(np.hstack(train_y))
    logger.info("Validation set: nights={}, epochs={}".format(
        len(valid_y), sum(len(y) for y in valid_y)))
    print_n_samples_each_class(np.hstack(valid_y))
    logger.info("Test set: nights={}, epochs={}".format(
        len(test_y), sum(len(y) for y in test_y)))
    print_n_samples_each_class(np.hstack(test_y))

    if bool(config.get("weighted_cross_ent", False)):
        config["class_weights"] = np.asarray(
            [1.0, 1.5, 1.0, 1.0, 1.0], dtype=np.float32)
    else:
        config["class_weights"] = np.ones(5, dtype=np.float32)
    logger.info("Class weights: {}".format(config["class_weights"]))
    logger.info("Signal augmentation: enabled={}, mode={}".format(
        config.get("augment_signal_full", False),
        config.get("signal_shift_mode", "within_epoch_zero")))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info("Using device {}".format(device))
    model = ModelV3(config=config, output_dir=fold_output_dir, device=device)
    n_parameters = sum(p.numel() for p in model.tsn.parameters())
    logger.info("Parameters: {}".format(n_parameters))

    best_valid_mf1 = -1.0
    selected_epoch = -1
    for epoch in range(int(config["n_epochs"])):
        shuffle_idx = np.random.permutation(np.arange(len(train_x)))
        augmented_x, augmented_y = _augment_training_data(
            train_x, train_y, config)
        train_outputs = model.train_with_dataloader(_make_minibatches(
            augmented_x,
            augmented_y,
            config,
            shuffle_idx=shuffle_idx,
            augment_seq=bool(config.get("augment_seq", False)),
        ))
        valid_outputs = model.evaluate_with_dataloader(_make_minibatches(
            valid_x, valid_y, config, shuffle_idx=None, augment_seq=False))

        writer = model.train_writer
        global_step = train_outputs["global_step"]
        writer.add_scalar(
            "e_losses/train", train_outputs["train/loss"], global_step)
        writer.add_scalar(
            "e_losses/transition",
            train_outputs["train/transition_loss"], global_step)
        writer.add_scalar(
            "e_losses/valid", valid_outputs["test/loss"], global_step)
        writer.add_scalar(
            "e_f1_score/train",
            train_outputs["train/f1_score"], global_step)
        writer.add_scalar(
            "e_f1_score/valid",
            valid_outputs["test/f1_score"], global_step)

        logger.info(
            "[e{}/{} s{}] TR (n={}) l={:.4f} tl={:.4f} "
            "a={:.1f} f1={:.1f} ({:.1f}s)| VA (n={}) l={:.4f} "
            "a={:.1f} f1={:.1f} ({:.1f}s)".format(
                epoch + 1,
                config["n_epochs"],
                global_step,
                len(train_outputs["train/trues"]),
                train_outputs["train/loss"],
                train_outputs["train/transition_loss"],
                train_outputs["train/accuracy"] * 100,
                train_outputs["train/f1_score"] * 100,
                train_outputs["train/duration"],
                len(valid_outputs["test/trues"]),
                valid_outputs["test/loss"],
                valid_outputs["test/accuracy"] * 100,
                valid_outputs["test/f1_score"] * 100,
                valid_outputs["test/duration"],
            ))

        if best_valid_mf1 < valid_outputs["test/f1_score"]:
            best_valid_mf1 = valid_outputs["test/f1_score"]
            selected_epoch = epoch + 1
            model.save_best_checkpoint()

        patience = config.get("no_improve_epochs")
        if (patience is not None and selected_epoch > 0
                and (epoch + 1 - selected_epoch) > int(patience)):
            logger.info("*** Early-stopping on validation Macro-F1 ***")
            break

    model.load_best_checkpoint()
    test_outputs = model.evaluate_with_dataloader(_make_minibatches(
        test_x, test_y, config, shuffle_idx=None, augment_seq=False))
    test_kappa = skmetrics.cohen_kappa_score(
        test_outputs["test/trues"], test_outputs["test/preds"])
    logger.info(
        "=== FINAL OUT-OF-FOLD TEST (validation-selected checkpoint) ===")
    logger.info("Fold {}: n={}, acc={:.2f}, mf1={:.2f}, kappa={:.4f}".format(
        fold_idx,
        len(test_outputs["test/trues"]),
        test_outputs["test/accuracy"] * 100,
        test_outputs["test/f1_score"] * 100,
        test_kappa,
    ))
    logger.info(">> Confusion Matrix")
    logger.info(test_outputs["test/cm"])

    np.savez(
        os.path.join(fold_output_dir, "fold_test_predictions.npz"),
        y_true=np.asarray(test_outputs["test/trues"]),
        y_pred=np.asarray(test_outputs["test/preds"]),
        fold_idx=fold_idx,
        train_sids=np.asarray(train_sids),
        valid_sids=np.asarray(valid_sids),
        test_sids=np.asarray(test_sids),
        best_valid_mf1=best_valid_mf1,
        selected_epoch=selected_epoch,
        test_kappa=test_kappa,
        model_variant=model_variant,
        n_parameters=n_parameters,
        context_length=int(config["seq_length"]),
        target_length=int(config["target_length"]),
        transition_loss_weight=(
            float(config.get("transition_loss_weight", 0.2))
            if model_variant == "v3_transition" else 0.0),
    )
    model.train_writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", required=True)
    parser.add_argument("--fold_idx", required=True, type=int)
    parser.add_argument("--output_dir", default="./results_v3/train")
    parser.add_argument("--log_file", default="./results_v3/fold.log")
    parser.add_argument("--model_variant", choices=MODEL_VARIANTS,
                        default="v3_context")
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--restart", action="store_true")
    args = parser.parse_args()
    train(
        config_file=args.config_file,
        fold_idx=args.fold_idx,
        output_dir=args.output_dir,
        log_file=args.log_file,
        model_variant=args.model_variant,
        restart=args.restart,
        random_seed=args.random_seed,
    )

