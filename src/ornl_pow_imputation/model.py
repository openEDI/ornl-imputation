from __future__ import annotations

import numpy as np
import torch
from torch import nn


def build_sequences(values: np.ndarray, sequence_length: int) -> tuple[np.ndarray, np.ndarray]:
    """Build causal windows where each target immediately follows its input window."""

    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError("values must be a two-dimensional array")
    if sequence_length <= 0:
        raise ValueError("sequence_length must be greater than zero")
    if len(array) <= sequence_length:
        return (
            np.empty((0, sequence_length, array.shape[1]), dtype=np.float32),
            np.empty((0, array.shape[1]), dtype=np.float32),
        )
    features = np.stack(
        [array[index : index + sequence_length] for index in range(len(array) - sequence_length)]
    )
    targets = array[sequence_length:]
    return features.astype(np.float32), targets.astype(np.float32)


class LSTMRegressor(nn.Module):
    """Causal many-to-one LSTM that predicts the next multi-phase sample."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        effective_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=effective_dropout,
        )
        self.output = nn.Linear(hidden_size, output_size)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        sequence, _ = self.lstm(inputs)
        return self.output(sequence[:, -1, :])
