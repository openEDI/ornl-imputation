from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ornl_pow_imputation import ImputerConfig, LSTMImputer

REFERENCE_DATA = Path(__file__).parents[1] / "examples" / "reference" / "data"
TARGET_COLUMNS = ("Voltage.Va", "Voltage.Vb", "Voltage.Vc")
SAMPLING_INTERVAL_GROUPS = (
    (
        "130048",
        ("sigId-242.csv", "sigId-243.csv", "sigId-245.csv", "sigId-246.csv"),
        "sigId-266.csv",
    ),
    ("65024", ("sigId-252.csv",), "sigId-253.csv"),
)


@pytest.mark.reference_quality
@pytest.mark.skipif(
    os.environ.get("RUN_REFERENCE_QUALITY") != "1",
    reason="set RUN_REFERENCE_QUALITY=1 to run held-out record benchmarks",
)
@pytest.mark.parametrize(
    ("interval_group", "training_files", "held_out_file"),
    SAMPLING_INTERVAL_GROUPS,
)
def test_held_out_record_rmse_beats_causal_forward_fill(
    interval_group: str, training_files: tuple[str, ...], held_out_file: str
) -> None:
    training = [pd.read_csv(REFERENCE_DATA / name) for name in training_files]
    truth = pd.read_csv(REFERENCE_DATA / held_out_file)
    config = ImputerConfig(
        sequence_length=32,
        hidden_size=32,
        num_layers=2,
        dropout=0.1,
        learning_rate=0.005,
        batch_size=128,
        max_epochs=40,
        patience=8,
        validation_fraction=0.1,
        test_fraction=0.1,
        max_gap=2,
        seed=42,
        device="cpu",
    )
    imputer = LSTMImputer(config)
    imputer.fit(training)

    incomplete = truth.copy()
    mask = pd.DataFrame(False, index=truth.index, columns=TARGET_COLUMNS)
    isolated_rows = np.arange(config.sequence_length + 8, len(truth), 32)
    mask.loc[isolated_rows, list(TARGET_COLUMNS)] = True
    for start in range(config.sequence_length + 24, len(truth) - 1, 128):
        mask.loc[start : start + 1, "Voltage.Va"] = True
        mask.loc[start : start + 1, "Voltage.Vc"] = True
    incomplete.loc[:, list(TARGET_COLUMNS)] = incomplete.loc[:, list(TARGET_COLUMNS)].mask(mask)

    predicted = imputer.impute(incomplete).data
    baseline = incomplete.copy()
    baseline.loc[:, list(TARGET_COLUMNS)] = baseline.loc[:, list(TARGET_COLUMNS)].ffill()

    phase_metrics: dict[str, dict[str, float]] = {}
    squared_model: list[np.ndarray] = []
    squared_baseline: list[np.ndarray] = []
    for column in TARGET_COLUMNS:
        selected = mask[column].to_numpy()
        expected = truth.loc[selected, column].to_numpy(dtype=np.float64)
        model_values = predicted.loc[selected, column].to_numpy(dtype=np.float64)
        baseline_values = baseline.loc[selected, column].to_numpy(dtype=np.float64)
        model_error = (model_values - expected) ** 2
        baseline_error = (baseline_values - expected) ** 2
        squared_model.append(model_error)
        squared_baseline.append(baseline_error)
        data_range = float(truth[column].max() - truth[column].min())
        model_rmse = float(np.sqrt(np.mean(model_error)))
        baseline_rmse = float(np.sqrt(np.mean(baseline_error)))
        phase_metrics[column] = {
            "model_rmse": model_rmse,
            "baseline_rmse": baseline_rmse,
            "model_nrmse_percent": 100.0 * model_rmse / data_range,
            "baseline_nrmse_percent": 100.0 * baseline_rmse / data_range,
        }

    model_rmse = float(np.sqrt(np.mean(np.concatenate(squared_model))))
    baseline_rmse = float(np.sqrt(np.mean(np.concatenate(squared_baseline))))
    print(
        json.dumps(
            {
                "sampling_interval_group": interval_group,
                "held_out_file": held_out_file,
                "masked_cells": int(mask.to_numpy().sum()),
                "aggregate_model_rmse": model_rmse,
                "aggregate_forward_fill_rmse": baseline_rmse,
                "phases": phase_metrics,
            },
            indent=2,
        )
    )
    assert model_rmse < baseline_rmse
