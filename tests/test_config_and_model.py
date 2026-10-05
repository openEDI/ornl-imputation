from __future__ import annotations

import numpy as np
import pytest
import torch

from ornl_pow_imputation import ImputerConfig
from ornl_pow_imputation.model import LSTMRegressor, build_sequences


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("sequence_length", 0),
        ("hidden_size", 0),
        ("num_layers", 0),
        ("batch_size", 0),
        ("max_epochs", 0),
        ("max_gap", 0),
        ("learning_rate", 0.0),
        ("dropout", -0.1),
        ("dropout", 1.0),
        ("validation_fraction", -0.1),
        ("test_fraction", 1.0),
    ],
)
def test_config_rejects_invalid_numeric_values(field: str, value: float) -> None:
    with pytest.raises(ValueError, match=field):
        ImputerConfig(**{field: value})


def test_config_rejects_empty_or_duplicate_target_columns() -> None:
    with pytest.raises(ValueError, match="target_columns"):
        ImputerConfig(target_columns=())
    with pytest.raises(ValueError, match="target_columns"):
        ImputerConfig(target_columns=("Voltage.Va", "Voltage.Va"))


def test_config_rejects_split_fractions_that_leave_no_training_data() -> None:
    with pytest.raises(ValueError, match="fraction"):
        ImputerConfig(validation_fraction=0.5, test_fraction=0.5)


def test_config_has_explicit_timestamp_and_cadence_defaults() -> None:
    config = ImputerConfig()

    assert config.timestamp_column == "Time µs"
    assert config.cadence_tolerance == pytest.approx(0.02)


@pytest.mark.parametrize("cadence_tolerance", [-0.01, 1.01])
def test_config_rejects_invalid_cadence_tolerance(cadence_tolerance: float) -> None:
    with pytest.raises(ValueError, match="cadence_tolerance"):
        ImputerConfig(cadence_tolerance=cadence_tolerance)


def test_build_sequences_has_expected_causal_shapes() -> None:
    values = np.arange(30, dtype=np.float32).reshape(10, 3)

    features, targets = build_sequences(values, sequence_length=4)

    assert features.shape == (6, 4, 3)
    assert targets.shape == (6, 3)
    np.testing.assert_array_equal(features[0], values[:4])
    np.testing.assert_array_equal(targets[0], values[4])
    np.testing.assert_array_equal(features[-1], values[-5:-1])
    np.testing.assert_array_equal(targets[-1], values[-1])


def test_lstm_forward_preserves_batch_and_target_dimensions() -> None:
    model = LSTMRegressor(
        input_size=3,
        hidden_size=8,
        output_size=3,
        num_layers=2,
        dropout=0.1,
    )

    output = model(torch.zeros((5, 4, 3), dtype=torch.float32))

    assert output.shape == (5, 3)
    assert torch.isfinite(output).all()
