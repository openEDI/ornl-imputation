from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ornl_pow_imputation import ImputerConfig

TARGET_COLUMNS = ("Voltage.Va", "Voltage.Vb", "Voltage.Vc")


def make_waveform(rows: int = 96) -> pd.DataFrame:
    """Return a small, repeatable three-phase point-on-wave data set."""
    samples = np.arange(rows, dtype=np.float32)
    angle = 2.0 * np.pi * samples / 16.0
    return pd.DataFrame(
        {
            "Time µs": np.arange(rows, dtype=np.int64) * 1_000,
            "Voltage.Va": np.sin(angle),
            "Voltage.Vb": np.sin(angle - 2.0 * np.pi / 3.0),
            "Voltage.Vc": np.sin(angle + 2.0 * np.pi / 3.0),
            "quality": np.arange(rows) % 3,
        }
    )


@pytest.fixture
def waveform() -> pd.DataFrame:
    return make_waveform()


@pytest.fixture
def tiny_config() -> ImputerConfig:
    return ImputerConfig(
        target_columns=TARGET_COLUMNS,
        sequence_length=4,
        hidden_size=8,
        num_layers=2,
        dropout=0.1,
        learning_rate=0.01,
        batch_size=16,
        max_epochs=3,
        patience=2,
        validation_fraction=0.2,
        test_fraction=0.2,
        max_gap=2,
        seed=42,
        device="cpu",
        warmup_strategy="last_observed_then_training_median",
    )
