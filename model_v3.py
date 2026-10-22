"""Training wrapper for boundary-aware Sleep-Mamba V3."""

import logging
import os
import timeit

import numpy as np
import sklearn.metrics as skmetrics
import torch
import torch.nn as nn
import torch.nn.functional as functional
from tensorboardX import SummaryWriter
from torch.optim import Adam

from network_v3 import TinySleepNetV3


logger = logging.getLogger("default_log")


class ModelV3:
    def __init__(self, config, output_dir, device):
        self.config = config
        self.output_dir = output_dir
        self.best_ckpt_path = os.path.join(output_dir, "best_ckpt")
        self.log_dir = os.path.join(output_dir, "log")
        self.device = device
        self.tsn = TinySleepNetV3(config).to(device)
        self.optimizer_all = Adam(
            self.tsn.parameters(),
            lr=config["learning_rate"],
            betas=(config["adam_beta_1"], config["adam_beta_2"]),
            eps=config["adam_epsilon"],
        )
        self.class_weights = torch.as_tensor(
            config["class_weights"], dtype=torch.float32, device=device)
        self.train_writer = SummaryWriter(os.path.join(
            self.log_dir, "train"))
        self.global_epoch = 0
        self.global_step = 0

    def get_current_epoch(self):
        return self.global_epoch

    def _classification_loss(self, logits, targets, target_mask):
        sample_losses = functional.cross_entropy(
            logits, targets.long(), reduction="none")
        sample_weights = self.class_weights.index_select(
            0, targets.long())
        weighted_mask = target_mask * sample_weights
        denominator = target_mask.sum().clamp_min(1.0)
        return (sample_losses * weighted_mask).sum() / denominator

    def _transition_loss(
            self, transition_logits, targets, target_mask, context_mask):
        if transition_logits is None:
            return targets.new_zeros((), dtype=torch.float32)

        batch_size = int(self.config["batch_size"])
        context_length = int(self.config["seq_length"])
        labels = targets.view(batch_size, context_length)
        target_positions = target_mask.view(batch_size, context_length)
        context_positions = context_mask.view(batch_size, context_length)

        transition_targets = (
            labels[:, 1:] != labels[:, :-1]).to(torch.float32)
        transition_mask = (
            target_positions[:, 1:]
            * context_positions[:, 1:]
            * context_positions[:, :-1]
        )
        positive_weight = torch.as_tensor(
            float(self.config.get("transition_pos_weight", 2.0)),
            device=self.device,
        )
        losses = functional.binary_cross_entropy_with_logits(
            transition_logits[:, 1:],
            transition_targets,
            reduction="none",
            pos_weight=positive_weight,
        )
        denominator = transition_mask.sum().clamp_min(1.0)
        return (losses * transition_mask).sum() / denominator

    def _cnn_regularization(self):
        regularization = torch.zeros((), device=self.device)
        for name, parameter in self.tsn.cnn.named_parameters():
            if "conv" in name:
                regularization = regularization + parameter.square().sum() / 2
        return float(self.config["l2_weight_decay"]) * regularization

    def _move_batch(self, batch):
        x, y, target_mask, context_mask, sequence_lengths, _ = batch
        batch_size = int(self.config["batch_size"])
        context_length = int(self.config["seq_length"])
        x = torch.from_numpy(x).view(
            batch_size * context_length, 1,
            int(self.config["input_size"])).to(self.device)
        y = torch.from_numpy(y).reshape(-1).to(self.device)
        target_mask = torch.from_numpy(target_mask).to(self.device)
        context_mask = torch.from_numpy(context_mask).to(self.device)
        return x, y, target_mask, context_mask, sequence_lengths

    @staticmethod
    def _append_predictions(logits, targets, target_mask, preds, trues):
        selected = target_mask.bool()
        batch_predictions = logits.argmax(dim=1)
        preds.extend(batch_predictions[selected].detach().cpu().tolist())
        trues.extend(targets[selected].detach().cpu().tolist())

    def train_with_dataloader(self, minibatches):
        self.tsn.train()
        start = timeit.default_timer()
        preds, trues, losses, transition_losses = [], [], [], []

        for batch in minibatches:
            x, y, target_mask, context_mask, _ = self._move_batch(batch)
            self.optimizer_all.zero_grad()
            logits, transition_logits = self.tsn(x, context_mask)
            classification_loss = self._classification_loss(
                logits, y, target_mask)
            transition_loss = self._transition_loss(
                transition_logits, y, target_mask, context_mask)
            total_loss = (
                classification_loss
                + float(self.config.get("transition_loss_weight", 0.2))
                * transition_loss
                + self._cnn_regularization()
            )
            total_loss.backward()
            nn.utils.clip_grad_norm_(
                self.tsn.parameters(),
                max_norm=float(self.config["clip_grad_value"]),
                norm_type=2,
            )
            self.optimizer_all.step()

            self._append_predictions(
                logits, y, target_mask, preds, trues)
            losses.append(float(total_loss.detach().cpu()))
            transition_losses.append(float(transition_loss.detach().cpu()))
            self.global_step += 1

        duration = timeit.default_timer() - start
        self.global_epoch += 1
        return {
            "global_step": self.global_step,
            "train/trues": trues,
            "train/preds": preds,
            "train/accuracy": skmetrics.accuracy_score(trues, preds),
            "train/loss": float(np.mean(losses)),
            "train/transition_loss": float(np.mean(transition_losses)),
            "train/f1_score": skmetrics.f1_score(
                trues, preds, average="macro"),
            "train/cm": skmetrics.confusion_matrix(
                trues, preds, labels=[0, 1, 2, 3, 4]),
            "train/duration": duration,
        }

    def evaluate_with_dataloader(self, minibatches):
        self.tsn.eval()
        start = timeit.default_timer()
        preds, trues, losses = [], [], []

        with torch.no_grad():
            for batch in minibatches:
                x, y, target_mask, context_mask, _ = self._move_batch(batch)
                logits, _ = self.tsn(x, context_mask)
                loss = self._classification_loss(logits, y, target_mask)
                self._append_predictions(
                    logits, y, target_mask, preds, trues)
                losses.append(float(loss.detach().cpu()))

        return {
            "test/trues": trues,
            "test/preds": preds,
            "test/loss": float(np.mean(losses)),
            "test/accuracy": skmetrics.accuracy_score(trues, preds),
            "test/f1_score": skmetrics.f1_score(
                trues, preds, average="macro"),
            "test/cm": skmetrics.confusion_matrix(
                trues, preds, labels=[0, 1, 2, 3, 4]),
            "test/duration": timeit.default_timer() - start,
        }

    def save_best_checkpoint(self, name="best_model"):
        os.makedirs(self.best_ckpt_path, exist_ok=True)
        save_path = os.path.join(
            self.best_ckpt_path, "{}.ckpt".format(name))
        torch.save(self.tsn.state_dict(), save_path)
        logger.info("Saved best checkpoint to {}".format(save_path))

    def load_best_checkpoint(self, name="best_model"):
        load_path = os.path.join(
            self.best_ckpt_path, "{}.ckpt".format(name))
        self.tsn.load_state_dict(torch.load(
            load_path, map_location=self.device, weights_only=True))
        logger.info(
            "Loaded validation-selected checkpoint from {}".format(
                load_path))

