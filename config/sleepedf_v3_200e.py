params = {
    # Training protocol: matched to the existing EDF-20 200-epoch runs.
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
    "l2_weight_decay": 1e-4,
    "model_variant": "v3_context",
    "mamba_d_state": 16,
    "mamba_d_conv": 4,
    "mamba_expand": 2,
    "spectral_n_fft": 256,
    "spectral_hop_length": 128,

    # Sleep-EDF-20: one subject per test fold.
    "dataset": "sleepedf",
    "data_dir": "D:/Zkaiwen/Tinysleepnet-pytorch/npz_data/eeg_fpz_cz",
    "n_folds": 20,
    "n_subjects": 20,

    # Fifty-epoch context; only the central twenty epochs are supervised.
    "seq_length": 50,
    "target_length": 20,
    "batch_size": 15,

    # Matched augmentation and class weighting.
    "augment_seq": True,
    "augment_signal_full": True,
    "signal_shift_percent": 0.1,
    "signal_shift_mode": "within_epoch_zero",
    "weighted_cross_ent": True,

    # Used only by v3_transition; ignored by v3_context.
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
