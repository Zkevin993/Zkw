"""Boundary-aware Sleep-Mamba V3 network.

V3 keeps the V1 CNN, spectral summary, projection width and BiMamba capacity.
The controlled change is mask-aware contextual temporal modelling.  An
optional transition head is available as a second, separately tested variant.
"""

from collections import OrderedDict

import torch
import torch.nn as nn

from mamba_block_v3 import MaskedBiMambaBlock


MODEL_VARIANTS = (
    "v3_context",
    "v3_transition",
    "v3_balanced_fusion",
    "v3_fixed_gate",
)


class TinySleepNetV3(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.variant = config.get("model_variant", "v3_context")
        if self.variant not in MODEL_VARIANTS:
            raise ValueError(
                "Unknown V3 variant {!r}; choose from {}".format(
                    self.variant, ", ".join(MODEL_VARIANTS)))

        padding_edf = {
            "conv1": (22, 22),
            "max_pool1": (2, 2),
            "conv2": (3, 4),
            "max_pool2": (0, 1),
        }
        first_filter_size = int(config["sampling_rate"] / 2.0)
        first_filter_stride = int(config["sampling_rate"] / 16.0)
        self.cnn = nn.Sequential(
            nn.ConstantPad1d(padding_edf["conv1"], 0),
            nn.Sequential(OrderedDict([
                ("conv1", nn.Conv1d(
                    1, 128, kernel_size=first_filter_size,
                    stride=first_filter_stride, bias=False)),
            ])),
            nn.BatchNorm1d(128, eps=0.001, momentum=0.01),
            nn.ReLU(inplace=True),
            nn.ConstantPad1d(padding_edf["max_pool1"], 0),
            nn.MaxPool1d(kernel_size=8, stride=8),
            nn.Dropout(p=0.5),
            nn.ConstantPad1d(padding_edf["conv2"], 0),
            nn.Sequential(OrderedDict([
                ("conv2", nn.Conv1d(
                    128, 128, kernel_size=8, stride=1, bias=False)),
            ])),
            nn.BatchNorm1d(128, eps=0.001, momentum=0.01),
            nn.ReLU(inplace=True),
            nn.ConstantPad1d(padding_edf["conv2"], 0),
            nn.Sequential(OrderedDict([
                ("conv3", nn.Conv1d(
                    128, 128, kernel_size=8, stride=1, bias=False)),
            ])),
            nn.BatchNorm1d(128, eps=0.001, momentum=0.01),
            nn.ReLU(inplace=True),
            nn.ConstantPad1d(padding_edf["conv2"], 0),
            nn.Sequential(OrderedDict([
                ("conv4", nn.Conv1d(
                    128, 128, kernel_size=8, stride=1, bias=False)),
            ])),
            nn.BatchNorm1d(128, eps=0.001, momentum=0.01),
            nn.ReLU(inplace=True),
            nn.ConstantPad1d(padding_edf["max_pool2"], 0),
            nn.MaxPool1d(kernel_size=4, stride=4),
            nn.Flatten(),
            nn.Dropout(p=0.5),
        )

        n_units = int(config["n_rnn_units"])
        n_fft = int(config.get("spectral_n_fft", 256))
        spectral_bins = n_fft // 2 + 1
        self.uses_balanced_fusion = self.variant in (
            "v3_balanced_fusion", "v3_fixed_gate")
        self.uses_learned_fusion_gate = (
            self.variant == "v3_balanced_fusion")

        if self.uses_balanced_fusion:
            # Project both feature sources into the same latent space before
            # fusion.  Independent normalization prevents either branch from
            # dominating only because of its dimensionality or value scale.
            self.time_projection = nn.Linear(2048, n_units)
            self.spectral_projection = nn.Linear(spectral_bins, n_units)
            self.time_norm = nn.LayerNorm(n_units)
            self.spectral_norm = nn.LayerNorm(n_units)
            if self.uses_learned_fusion_gate:
                self.fusion_gate_logits = nn.Parameter(torch.zeros(n_units))
            self.fusion_norm = nn.LayerNorm(n_units)
            self.fusion_activation = nn.GELU()
        else:
            # Preserve the exact V3 projection for the existing controlled
            # context and transition experiments.
            self.feature_projection = nn.Linear(
                2048 + spectral_bins, n_units)
        self.temporal = MaskedBiMambaBlock(
            d_model=n_units,
            d_state=int(config.get("mamba_d_state", 16)),
            d_conv=int(config.get("mamba_d_conv", 4)),
            expand=int(config.get("mamba_expand", 2)),
        )
        self.output_dropout = nn.Dropout(p=0.5)
        self.classifier = nn.Linear(n_units, int(config["n_classes"]))
        self.transition_head = (
            nn.Linear(n_units, 1)
            if self.variant == "v3_transition" else None
        )

        self.n_fft = n_fft
        self.spectral_hop_length = int(
            config.get("spectral_hop_length", 128))
        self.register_buffer("spectral_window", torch.hann_window(n_fft))

    def _spectral_features(self, x):
        x_raw = x.squeeze(1)
        spectrum = torch.stft(
            x_raw,
            n_fft=self.n_fft,
            hop_length=self.spectral_hop_length,
            window=self.spectral_window.to(dtype=x_raw.dtype),
            return_complex=True,
        )
        # Deliberately preserve V1's spectral representation in v3_context so
        # the context correction is tested as a single controlled change.
        return spectrum.abs().mean(dim=-1)

    def forward(self, x, context_mask):
        batch_size = int(self.config["batch_size"])
        context_length = int(self.config["seq_length"])
        expected = batch_size * context_length
        if x.shape[0] != expected:
            raise ValueError(
                "expected {} epochs, received {}".format(expected, x.shape[0]))

        mask = context_mask.reshape(batch_size, context_length).bool()
        flat_mask = mask.reshape(-1)
        valid_indices = torch.nonzero(flat_mask, as_tuple=False).squeeze(1)
        if valid_indices.numel() == 0:
            raise ValueError("a V3 batch must contain at least one real epoch")

        # Encode only real epochs so zero padding cannot alter BatchNorm stats.
        valid_x = x.index_select(0, valid_indices)
        time_features = self.cnn(valid_x).view(-1, 2048)
        spectral_features = self._spectral_features(valid_x)
        if self.uses_balanced_fusion:
            # Magnitude spectra are strongly right-skewed.  log1p preserves
            # ordering while limiting the influence of unusually large bins.
            spectral_features = torch.log1p(spectral_features)
            time_latent = self.fusion_activation(
                self.time_norm(self.time_projection(time_features)))
            spectral_latent = self.fusion_activation(
                self.spectral_norm(
                    self.spectral_projection(spectral_features)))
            gate = (
                torch.sigmoid(self.fusion_gate_logits).unsqueeze(0)
                if self.uses_learned_fusion_gate else 0.5
            )
            valid_features = self.fusion_norm(
                gate * time_latent + (1.0 - gate) * spectral_latent)
        else:
            valid_features = self.feature_projection(torch.cat(
                [time_features, spectral_features], dim=-1))

        all_features = valid_features.new_zeros((
            expected, valid_features.shape[-1]))
        all_features = all_features.index_copy(
            0, valid_indices, valid_features)
        sequence = all_features.view(
            batch_size, context_length, -1)
        sequence = self.temporal(sequence, mask)

        flat_sequence = sequence.reshape(-1, sequence.shape[-1])
        class_logits = self.classifier(self.output_dropout(flat_sequence))
        transition_logits = None
        if self.transition_head is not None:
            transition_logits = self.transition_head(sequence).squeeze(-1)
        return class_logits, transition_logits


if __name__ == "__main__":
    from config.sleepedfx_v3_200e import train

    for variant in MODEL_VARIANTS:
        variant_config = train.copy()
        variant_config["model_variant"] = variant
        model = TinySleepNetV3(variant_config)
        n_parameters = sum(p.numel() for p in model.parameters())
        print(variant, n_parameters)
