params = {
    # Training protocol: unchanged from the 200-epoch EDF-78 runs.
    "n_epochs": 200,
    "learning_rate": 1e-4,
    "adam_beta_1": 0.9,
    "adam_beta_2": 0.999,
    "adam_epsilon": 1e-8,
    "clip_grad_value": 5.0,
    "evaluate_span": 50,
    "checkpoint_span": 50,
    "no_improve_epochs": 50,

    # V1-sized feature extractor and BiMamba.
    "model": "sleep-mamba-v3",
    "n_rnn_layers": 1,
    "n_rnn_units": 128,
    "sampling_rate": 100.0,
    "input_size": 3000,
    "n_classes": 5,
    "l2_weight_decay": 1e-3,
    "model_variant": "v3_context",
    "mamba_d_state": 16,
    "mamba_d_conv": 4,
    "mamba_expand": 2,
    "spectral_n_fft": 256,
    "spectral_hop_length": 128,

    # Sleep-EDF Expanded (Sleep Cassette): 78 subjects, 153 recordings.
    "dataset": "sleepedfx",
    "data_dir": "D:/Zkaiwen/SleepEDF78/npz/eeg_fpz_cz",
    "n_folds": 10,
    "n_subjects": 78,

    # Context windows supervise 20 central epochs and use 15 epochs of
    # additional context on each side whenever the recording permits it.
    "seq_length": 50,
    "target_length": 20,
    "batch_size": 15,

    # Keep the current controlled augmentation and class weighting protocol.
    "augment_seq": True,
    "augment_signal_full": True,
    "signal_shift_percent": 0.1,
    "signal_shift_mode": "within_epoch_zero",
    "weighted_cross_ent": True,

    # Used only by v3_transition; v3_context ignores these values.
    "transition_loss_weight": 0.2,
    "transition_pos_weight": 2.0,
}

train = params.copy()

predict = params.copy()
predict.update({
    "batch_size": 1,
    "seq_length": 50,
    "target_length": 20,
})
