"""Context-centred minibatches for Sleep-Mamba V3.

Every real epoch is a supervised target exactly once.  A target block is
surrounded by additional past and future epochs that are used only as context.
This removes hard Mamba boundaries without duplicating samples in the metrics.
"""

import math
import numpy as np


def _make_windows(
        inputs, targets, context_length, target_length,
        sequence_indices, augment_seq):
    if context_length < target_length:
        raise ValueError("context_length must be >= target_length")
    if context_length <= 0 or target_length <= 0:
        raise ValueError("sequence lengths must be positive")

    left_context = (context_length - target_length) // 2
    right_context = context_length - target_length - left_context
    windows = []

    for night_index in sequence_indices:
        night_x = np.asarray(inputs[int(night_index)])
        night_y = np.asarray(targets[int(night_index)])
        if len(night_x) != len(night_y):
            raise ValueError("input and target night lengths do not match")

        # Match the original sequence augmentation: randomly omit up to four
        # leading epochs during training only.
        if augment_seq and len(night_y) > 1:
            n_skips = int(np.random.randint(5))
            night_x = night_x[n_skips:]
            night_y = night_y[n_skips:]

        n_epochs = len(night_y)
        for target_start in range(0, n_epochs, target_length):
            target_stop = min(target_start + target_length, n_epochs)
            nominal_start = target_start - left_context
            nominal_stop = target_stop + right_context
            source_start = max(0, nominal_start)
            source_stop = min(n_epochs, nominal_stop)
            destination_start = source_start - nominal_start
            destination_stop = destination_start + source_stop - source_start
            target_destination_start = target_start - nominal_start
            target_destination_stop = (
                target_destination_start + target_stop - target_start)

            windows.append({
                "x": night_x[source_start:source_stop],
                "y": night_y[source_start:source_stop],
                "destination_start": destination_start,
                "destination_stop": destination_stop,
                "target_start": target_destination_start,
                "target_stop": target_destination_stop,
            })

    return windows


def iterate_context_minibatches(
        inputs, targets, batch_size, context_length, target_length,
        shuffle_idx=None, augment_seq=False):
    """Yield fixed context windows and separate context/target masks."""

    if len(inputs) != len(targets):
        raise ValueError("inputs and targets must contain the same nights")
    if not inputs:
        return

    if shuffle_idx is None:
        sequence_indices = np.arange(len(inputs))
    else:
        sequence_indices = np.asarray(shuffle_idx)

    windows = _make_windows(
        inputs=inputs,
        targets=targets,
        context_length=context_length,
        target_length=target_length,
        sequence_indices=sequence_indices,
        augment_seq=augment_seq,
    )
    if shuffle_idx is not None:
        np.random.shuffle(windows)

    input_shape = np.asarray(inputs[0]).shape[1:]
    target_shape = np.asarray(targets[0]).shape[1:]
    n_batches = int(math.ceil(len(windows) / float(batch_size)))

    for batch_index in range(n_batches):
        selected = windows[
            batch_index * batch_size:(batch_index + 1) * batch_size]
        batch_x = np.zeros(
            (batch_size, context_length) + input_shape, dtype=np.float32)
        batch_y = np.zeros(
            (batch_size, context_length) + target_shape, dtype=np.int64)
        context_mask = np.zeros(
            (batch_size, context_length), dtype=np.float32)
        target_mask = np.zeros(
            (batch_size, context_length), dtype=np.float32)

        for row, window in enumerate(selected):
            d0 = window["destination_start"]
            d1 = window["destination_stop"]
            t0 = window["target_start"]
            t1 = window["target_stop"]
            batch_x[row, d0:d1] = window["x"]
            batch_y[row, d0:d1] = window["y"]
            context_mask[row, d0:d1] = 1.0
            target_mask[row, t0:t1] = 1.0

        # The final boolean is kept for interface clarity; V3 windows are
        # independent and never carry an LSTM state between batches.
        yield (
            batch_x.reshape((-1,) + input_shape),
            batch_y.reshape((-1,) + target_shape),
            target_mask.reshape(-1),
            context_mask.reshape(-1),
            context_mask.sum(axis=1).astype(np.int64),
            True,
        )

