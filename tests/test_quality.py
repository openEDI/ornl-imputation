from __future__ import annotations

import os

import numpy as np
import pytest
from conftest import TARGET_COLUMNS, make_waveform

from ornl_pow_imputation import ImputerConfig, LSTMImputer


@pytest.mark.slow
@pytest.mark.skipif(
    os.environ.get("RUN_SLOW_QUALITY") != "1",
    reason="set RUN_SLOW_QUALITY=1 to run the model-quality acceptance check",
)
def test_missing_point_rmse_beats_causal_forward_fill() -> None:
    complete = make_waveform(rows=512)
    train = complete.iloc[:384].reset_index(drop=True)
    held_out = complete.iloc[384:].reset_index(drop=True)
    incomplete = held_out.copy()
    # Keep the acceptance mask beyond the causal warm-up window so fallback
    # values are never counted as LSTM predictions.
    missing_rows = np.arange(24, len(incomplete), 8)
    incomplete.loc[missing_rows, list(TARGET_COLUMNS)] = np.nan
    config = ImputerConfig(
        target_columns=TARGET_COLUMNS,
        sequence_length=16,
        hidden_size=32,
        num_layers=2,
        dropout=0.1,
        learning_rate=0.005,
        batch_size=32,
        max_epochs=100,
        patience=12,
        validation_fraction=0.15,
        test_fraction=0.15,
        max_gap=2,
        seed=42,
        device="cpu",
    )
    imputer = LSTMImputer(config)
    imputer.fit([train])

    predicted = imputer.impute(incomplete).data
    forward_fill = incomplete.copy()
    forward_fill[list(TARGET_COLUMNS)] = forward_fill[list(TARGET_COLUMNS)].ffill()
    truth = held_out.loc[missing_rows, list(TARGET_COLUMNS)].to_numpy()
    model_values = predicted.loc[missing_rows, list(TARGET_COLUMNS)].to_numpy()
    baseline_values = forward_fill.loc[missing_rows, list(TARGET_COLUMNS)].to_numpy()
    model_rmse = float(np.sqrt(np.mean((model_values - truth) ** 2)))
    baseline_rmse = float(np.sqrt(np.mean((baseline_values - truth) ** 2)))

    assert model_rmse < baseline_rmse, (
        f"LSTM RMSE {model_rmse:.6f} did not beat causal forward-fill RMSE {baseline_rmse:.6f}"
    )
