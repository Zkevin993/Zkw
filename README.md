# BTF-Net

Code for **Efficient Single-Channel EEG Sleep Staging via Boundary-Aware Context and Balanced Time-Frequency Fusion**.

Jeng-Shyang Pan, Kaiwen Zheng, Hao Zhou, Shu-Chuan Chu, and Václav Snášel

BTF-Net performs offline sleep staging from a single Fpz-Cz EEG channel. It combines a convolutional waveform encoder with an STFT spectral branch, projects both representations into a shared feature space, and fuses them with learnable element-wise weights. A bidirectional Mamba module models neighboring sleep epochs. Overlapping context windows and two masks distinguish real input, auxiliary context, and classification targets.

The final model is selected with `--model_variant v3_balanced_fusion`. The implementation retains the experimental `_v3` filenames.

## Model

| Component | Setting |
| --- | --- |
| EEG input | Fpz-Cz, 100 Hz, 30 seconds per epoch, 3,000 samples |
| Context window | 50 epochs: 15 left-context positions, 20 target positions, and 15 right-context positions |
| Window stride | 20 epochs during validation and testing |
| CNN representation | 2,048 features per real epoch |
| Spectral representation | STFT with a 256-sample Hann window, 256-point FFT, and 128-sample hop; mean magnitude over time frames |
| Balanced fusion | Independent 128-dimensional projections and normalization; learnable sigmoid gate initialized to equal weighting |
| Temporal module | Bidirectional Mamba; model dimension 128, state dimension 16, convolution width 4, expansion factor 2 |
| Classes | W, N1, N2, N3, REM |

The valid-position mask identifies real epochs and controls feature encoding and temporal reordering. The target mask selects the real target epochs used for supervision and evaluation. Each retained epoch contributes once to the test predictions.

## Reported results

The manuscript uses subject-independent cross-validation: 20 folds on EDF-20 and 10 folds on EDF-78.

| Dataset | Accuracy (%) | Macro-F1 (%) | Cohen's kappa | N1 F1 (%) |
| --- | ---: | ---: | ---: | ---: |
| EDF-20 | 83.92 | 79.11 | 0.7826 | 52.62 |
| EDF-78 | 82.37 | 77.49 | 0.7580 | 50.40 |

These scores are computed from the combined out-of-fold predictions. The fold-wise Macro-F1 means and sample standard deviations are **78.22 ± 7.81%** on EDF-20 and **76.43 ± 4.92%** on EDF-78.

BTF-Net has approximately **0.947 million trainable parameters**. Its reported median forward-pass latency is **4.147 ms per 50-epoch input sequence**, measured on an NVIDIA RTX 4090 D with PyTorch 2.4.0, CUDA 12.1, FP32 precision, and batch size 1. The reported throughput of **12,056.3 input epochs/s** counts all 50 input positions, including auxiliary context. Timing covers the network forward pass with inputs already on the GPU.

## Repository layout

Place the experiment files in the following layout. The shared data-loading, logging, utility, preprocessing, and aggregation files must accompany the model files.

```text
BTF-Net/
├── README.md
├── LICENSE
├── network_v3.py
├── mamba_block_v3.py
├── minibatching_v3.py
├── model_v3.py
├── train_v3.py
├── run_v3_ablation.py
├── benchmark_efficiency_v3.py
├── data.py
├── logger.py
├── utils.py
├── aggregate_folds.py
├── sleepedf.txt
├── sleepedfx.txt
├── config/
│   ├── sleepedf_v3_200e.py
│   └── sleepedfx_v3_200e.py
└── script/                         # Dataset preprocessing scripts
```

## Environment

The reported GPU experiments used **PyTorch 2.4.0** and **CUDA 12.1**. Training also requires NumPy, scikit-learn, tensorboardX, and a compatible installation of `mamba-ssm`. Install the CUDA-enabled PyTorch build first, then install the remaining packages in the same environment:

```bash
python -m pip install numpy scikit-learn tensorboardX tensorboard
```

Install the Mamba implementation and any required `causal-conv1d` extension for the chosen Python, PyTorch, CUDA, and operating-system combination. This code imports the Mamba-1 block using `from mamba_ssm import Mamba`. Refer to the [Mamba installation instructions](https://github.com/state-spaces/mamba#installation) for the appropriate build. Preserve the versions and any platform-specific modifications used in the successful training environment when reproducing the reported results.

Check the main imports before starting an experiment:

```bash
python -c "import torch, numpy, sklearn, tensorboardX; from mamba_ssm import Mamba; print('PyTorch:', torch.__version__); print('CUDA:', torch.version.cuda); print('GPU available:', torch.cuda.is_available())"
```

Additional preprocessing dependencies are determined by the EDF-reading scripts. Run the commands below from the repository root in the configured environment.

## Data preparation

Download the Sleep Cassette recordings and annotations from the [PhysioNet Sleep-EDF Expanded database](https://physionet.org/content/sleep-edfx/1.0.0/).

The manuscript uses the following preprocessing settings:

- Select the Fpz-Cz EEG channel at its original sampling rate of 100 Hz.
- Divide each recording into non-overlapping 30-second epochs.
- Map labels to W, N1, N2, N3, and REM, merging stages 3 and 4 into N3.
- Remove movement and unscored epochs; retain at most 30 minutes of wake before and after the sleep interval.
- Preserve recording order and recording boundaries when exporting the preprocessed NPZ files.

The resulting subsets contain 20 subjects and 39 recordings for EDF-20, and 78 subjects and 153 recordings for EDF-78. Their retained epoch counts are 44,220 and 199,352, respectively. EDF-20 is a subset of EDF-78.

Set `data_dir` in each configuration file to the corresponding folder of preprocessed NPZ recordings. For example:

```python
"data_dir": "/path/to/preprocessed/edf78/eeg_fpz_cz",
```

Keep `sleepedf.txt` and `sleepedfx.txt` in the repository root. Each line must contain one integer subject identifier. The subject order in these files determines the test folds; retain the subject identifiers and order used for the reported experiments. The training script keeps all recordings from the same subject in the same split and selects validation subjects from the remaining subjects using the specified random seed.

## Training and evaluation

### Check one fold

Run EDF-78 fold 0 before launching the full cross-validation:

```bash
python run_v3_ablation.py --model_variant v3_balanced_fusion --config_file config/sleepedfx_v3_200e.py --results_dir results/edf78 --from_fold 0 --to_fold 0 --random_seed 42 --gpu 0 --skip_completed
```

`--gpu 0` selects physical GPU 0. Change the index to use another GPU. The selected device appears as `cuda:0` inside the training process.

### EDF-20: 20-fold cross-validation

```bash
python run_v3_ablation.py --model_variant v3_balanced_fusion --config_file config/sleepedf_v3_200e.py --results_dir results/edf20 --from_fold 0 --to_fold 19 --random_seed 42 --gpu 0 --skip_completed
```

### EDF-78: 10-fold cross-validation

```bash
python run_v3_ablation.py --model_variant v3_balanced_fusion --config_file config/sleepedfx_v3_200e.py --results_dir results/edf78 --from_fold 0 --to_fold 9 --random_seed 42 --gpu 0 --skip_completed
```

The configurations use a maximum of 200 training epochs, a batch size of 15 windows, and a learning rate of `1e-4`. Each fold selects its checkpoint using validation Macro-F1 and evaluates the selected checkpoint on the test subjects. The default random seed is 42.

With `--skip_completed`, folds containing `fold_test_predictions.npz` are retained. Unfinished folds restart training from scratch. Once every fold is complete, the runner calls `aggregate_folds.py` and writes the combined results to `cv_results.npz`.

For example, the EDF-78 output layout is:

```text
results/edf78/v3_balanced_fusion/
├── train/
│   ├── 0/fold_test_predictions.npz
│   ├── 1/fold_test_predictions.npz
│   └── ...
├── logs/
│   ├── fold_0.log
│   └── ...
└── cv_results.npz
```

Per-fold prediction files include labels, predictions, subject splits, the selected training epoch, and experiment metadata.

## Fusion ablations

| Command-line variant | Configuration |
| --- | --- |
| `v3_context` | Boundary-aware context and masked bidirectional Mamba with concatenated CNN/STFT features followed by a joint projection |
| `v3_balanced_fusion` | Final BTF-Net with independent branch projections, normalization, and learnable element-wise fusion |
| `v3_fixed_gate` | BTF-Net with fusion weights fixed to 0.5 |

Run a variant by changing `--model_variant` in the training commands. Each variant has a separate output subdirectory.

## Efficiency benchmark

```bash
python benchmark_efficiency_v3.py --config_file config/sleepedfx_v3_200e.py --variants v3_context v3_balanced_fusion v3_fixed_gate --device cuda --precision fp32 --batch_size 1 --seq_length 50 --warmup 30 --repetitions 100 --output results/efficiency_v3.csv
```

The benchmark writes CSV results and a JSON file containing runtime metadata. It uses synthetic GPU-resident inputs, evaluation mode, and disabled gradient computation. GPU measurements use CUDA events. The throughput is calculated from input epochs per forward pass divided by the median latency.

## Offline inference

BTF-Net uses context on both sides of the target block. A full target block requires 15 subsequent epochs, equivalent to 7.5 minutes of look-ahead after its final target epoch. The current design supports retrospective full-night staging; its GPU forward-pass latency and context acquisition delay describe different parts of the processing pipeline.

## Citation

The associated manuscript is:

> Jeng-Shyang Pan, Kaiwen Zheng, Hao Zhou, Shu-Chuan Chu, and Václav Snášel. *Efficient Single-Channel EEG Sleep Staging via Boundary-Aware Context and Balanced Time-Frequency Fusion*. Manuscript, 2026.

```bibtex
@unpublished{pan2026btfnet,
  title = {Efficient Single-Channel EEG Sleep Staging via Boundary-Aware Context and Balanced Time-Frequency Fusion},
  author = {Pan, Jeng-Shyang and Zheng, Kaiwen and Zhou, Hao and Chu, Shu-Chuan and Sn{\'{a}}{\v{s}}el, V{\'{a}}clav},
  year = {2026},
  note = {Manuscript}
}
```

## Acknowledgments and license

The waveform encoder follows the TinySleepNet architecture. We acknowledge [TinySleepNet](https://github.com/akaraspt/tinysleepnet) and [Mamba](https://github.com/state-spaces/mamba), both released under Apache-2.0 in their original repositories.

Original BTF-Net contributions are licensed under the **Apache License 2.0**; see [LICENSE](LICENSE). Third-party code retains its original copyright and license terms. Any inherited PyTorch implementation files must retain their upstream notices, and their redistribution permissions must be verified against the actual source repository. The project license does not grant rights to third-party code for which permission has not been established. Sleep-EDF recordings remain subject to the terms on PhysioNet.
