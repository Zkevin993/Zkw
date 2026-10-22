"""Mask-aware Mamba blocks used by the isolated V3 experiments."""

import torch
import torch.nn as nn
from mamba_ssm import Mamba


class MaskedBiMambaBlock(nn.Module):
    """Bidirectional Mamba that excludes padded positions from both scans.

    The V3 minibatcher guarantees that valid context positions are contiguous
    within every row.  We first pack each valid interval to the beginning of
    the row, run causal forward and reversed scans, then unpack the result.
    Padding therefore always appears *after* valid positions for both causal
    scans and cannot contaminate a real epoch representation.
    """

    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.forward_mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.backward_mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.fusion = nn.Linear(d_model * 2, d_model)
        self.layer_norm = nn.LayerNorm(d_model)

    @staticmethod
    def _gather(sequence, indices):
        return sequence.gather(
            1, indices.unsqueeze(-1).expand(-1, -1, sequence.shape[-1]))

    def forward(self, sequence, valid_mask):
        if sequence.ndim != 3:
            raise ValueError("sequence must have shape [batch, length, dim]")
        if valid_mask.shape != sequence.shape[:2]:
            raise ValueError("valid_mask must match [batch, length]")

        mask = valid_mask.to(dtype=torch.bool)
        batch_size, sequence_length, _ = sequence.shape
        positions = torch.arange(
            sequence_length, device=sequence.device).unsqueeze(0)
        positions = positions.expand(batch_size, -1)

        lengths = mask.sum(dim=1)
        starts = mask.to(dtype=torch.int64).argmax(dim=1)
        starts = torch.where(lengths > 0, starts, torch.zeros_like(starts))

        # Pack every contiguous valid interval into a valid prefix.
        packed_indices = (starts.unsqueeze(1) + positions).clamp(
            max=sequence_length - 1)
        prefix_mask = positions < lengths.unsqueeze(1)
        prefix_scale = prefix_mask.unsqueeze(-1).to(sequence.dtype)
        packed = self._gather(sequence, packed_indices) * prefix_scale

        forward_output = self.forward_mamba(packed) * prefix_scale

        # Reverse only the valid prefix, never the padded tail.
        reverse_indices = (
            lengths.unsqueeze(1) - 1 - positions).clamp(
                min=0, max=sequence_length - 1)
        reversed_packed = self._gather(packed, reverse_indices) * prefix_scale
        backward_reversed = self.backward_mamba(reversed_packed) * prefix_scale
        backward_output = (
            self._gather(backward_reversed, reverse_indices) * prefix_scale)

        fused = self.fusion(torch.cat(
            [forward_output, backward_output], dim=-1))
        fused = self.layer_norm(fused + packed) * prefix_scale

        # Map packed representations back to their original context positions.
        unpack_indices = (positions - starts.unsqueeze(1)).clamp(
            min=0, max=sequence_length - 1)
        unpacked = self._gather(fused, unpack_indices)
        return unpacked * mask.unsqueeze(-1).to(sequence.dtype)

