from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

from ornl_pow_imputation import (
    ArtifactError,
    GapTooLongError,
    ImputerConfig,
    InsufficientContextError,
    LSTMImputer,
    SchemaError,
)


def test_fit_is_deterministic_for_a_fixed_seed(waveform, tiny_config) -> None:
    first = LSTMImputer(tiny_config)
    second = LSTMImputer(tiny_config)

    first_report = first.fit([waveform])
    second_report = second.fit([waveform])

    assert first_report.training_loss == pytest.approx(second_report.training_loss)
    assert first_report.validation_loss == pytest.approx(second_report.validation_loss)
    for first_parameter, second_parameter in zip(
        first.model.parameters(), second.model.parameters(), strict=True
    ):
        np.testing.assert_array_equal(
            first_parameter.detach().cpu().numpy(),
            second_parameter.detach().cpu().numpy(),
        )


def test_impute_preserves_observed_values_and_non_target_columns(waveform, tiny_config) -> None:
    imputer = LSTMImputer(tiny_config)
    imputer.fit([waveform])
    incomplete = waveform.copy()
    incomplete.loc[20, "Voltage.Va"] = np.nan
    incomplete.loc[40:41, "Voltage.Vb"] = np.nan

    result = imputer.impute(incomplete)

    assert pd.notna(result.data.loc[20, "Voltage.Va"])
    assert result.data.loc[40:41, "Voltage.Vb"].notna().all()
    pdt.assert_series_equal(result.data["Time µs"], incomplete["Time µs"])
    pdt.assert_series_equal(result.data["quality"], incomplete["quality"])
    observed = incomplete[list(tiny_config.target_columns)].notna()
    pdt.assert_frame_equal(
        result.data[list(tiny_config.target_columns)].where(observed),
        incomplete[list(tiny_config.target_columns)].where(observed),
        check_exact=True,
    )


def test_imputed_mask_marks_only_original_target_nans(waveform, tiny_config) -> None:
    imputer = LSTMImputer(tiny_config)
    imputer.fit([waveform])
    incomplete = waveform.copy()
    incomplete.loc[10, "Voltage.Va"] = np.nan
    incomplete.loc[11, "Voltage.Vc"] = np.nan
    incomplete.loc[12, "quality"] = np.nan

    result = imputer.impute(incomplete)
    expected = incomplete[list(tiny_config.target_columns)].isna()

    pdt.assert_frame_equal(result.imputed_mask, expected)
    assert tuple(result.data.columns) == tuple(incomplete.columns)
    assert pd.isna(result.data.loc[12, "quality"])


@pytest.mark.parametrize("missing_column", ["Voltage.Va", "Voltage.Vb", "Voltage.Vc"])
def test_schema_error_names_a_missing_target_column(waveform, tiny_config, missing_column) -> None:
    imputer = LSTMImputer(tiny_config)

    with pytest.raises(SchemaError, match=missing_column):
        imputer.fit([waveform.drop(columns=missing_column)])


def test_schema_error_rejects_non_numeric_target(waveform, tiny_config) -> None:
    invalid = waveform.copy()
    invalid["Voltage.Va"] = "not-a-number"

    with pytest.raises(SchemaError, match="Voltage.Va"):
        LSTMImputer(tiny_config).fit([invalid])


@pytest.mark.parametrize("value", [float("inf"), float("-inf")])
def test_schema_error_rejects_non_finite_target_with_row_evidence(
    waveform, tiny_config, value
) -> None:
    invalid_training = waveform.copy()
    invalid_training.loc[12, "Voltage.Va"] = value
    with pytest.raises(SchemaError, match=r"Voltage\.Va.*row 12"):
        LSTMImputer(tiny_config).fit([invalid_training])

    imputer = LSTMImputer(tiny_config)
    imputer.fit([waveform])
    invalid_input = waveform.copy()
    invalid_input.loc[15, "Voltage.Vb"] = value
    with pytest.raises(SchemaError, match=r"Voltage\.Vb.*row 15"):
        imputer.impute(invalid_input)


def test_save_load_round_trip_preserves_predictions(tmp_path, waveform, tiny_config) -> None:
    imputer = LSTMImputer(tiny_config)
    imputer.fit([waveform])
    incomplete = waveform.copy()
    incomplete.loc[24, "Voltage.Va"] = np.nan
    before = imputer.impute(incomplete)
    artifact = tmp_path / "pow-lstm"

    imputer.save(artifact)
    metadata_text = (artifact / "metadata.json").read_text(encoding="utf-8")
    metadata = json.loads(metadata_text)
    assert metadata["artifact_schema_version"] == 1
    assert metadata["model_metadata"]["artifact_schema_version"] == 1
    assert metadata["model_metadata"]["deterministic_algorithms"] is True
    assert metadata["model_metadata"]["runtime_versions"]["package"] == "0.1.0"
    assert "NaN" not in metadata_text
    restored = LSTMImputer.load(artifact, device="cpu")
    after = restored.impute(incomplete)

    pdt.assert_frame_equal(after.data, before.data, check_exact=True)
    pdt.assert_frame_equal(after.imputed_mask, before.imputed_mask)
    assert after.model_metadata == before.model_metadata


def test_load_rejects_unknown_artifact_schema(tmp_path, waveform, tiny_config) -> None:
    artifact = tmp_path / "pow-lstm"
    imputer = LSTMImputer(tiny_config)
    imputer.fit([waveform])
    imputer.save(artifact)
    metadata_path = artifact / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["artifact_schema_version"] = 999
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ArtifactError, match="schema version 999"):
        LSTMImputer.load(artifact)


def test_short_consecutive_gap_is_filled_autoregressively(waveform, tiny_config) -> None:
    imputer = LSTMImputer(tiny_config)
    imputer.fit([waveform])
    incomplete = waveform.copy()
    incomplete.loc[32:33, "Voltage.Va"] = np.nan

    result = imputer.impute(incomplete)

    assert result.data.loc[32:33, "Voltage.Va"].notna().all()
    assert result.imputed_mask.loc[32:33, "Voltage.Va"].all()
    assert result.gap_diagnostics.max_gap_by_column["Voltage.Va"] == 2


def test_gap_longer_than_configured_limit_is_rejected(waveform, tiny_config) -> None:
    imputer = LSTMImputer(tiny_config)
    imputer.fit([waveform])
    incomplete = waveform.copy()
    incomplete.loc[32:34, "Voltage.Va"] = np.nan

    with pytest.raises(GapTooLongError, match=r"Voltage\.Va.*3.*2"):
        imputer.impute(incomplete)


def test_warmup_gap_uses_last_observed_then_training_median(waveform, tiny_config) -> None:
    imputer = LSTMImputer(tiny_config)
    imputer.fit([waveform])
    incomplete = waveform.copy()
    incomplete.loc[0, "Voltage.Va"] = np.nan
    incomplete.loc[2, "Voltage.Vb"] = np.nan

    result = imputer.impute(incomplete)

    assert result.data.loc[0, "Voltage.Va"] == pytest.approx(imputer.training_medians["Voltage.Va"])
    assert result.data.loc[2, "Voltage.Vb"] == pytest.approx(incomplete.loc[1, "Voltage.Vb"])
    assert result.fallback_used.loc[0, "Voltage.Va"] == "training_median"
    assert result.fallback_used.loc[2, "Voltage.Vb"] == "last_observed"
    assert result.provenance.loc[0, "Voltage.Va"] == "training_median"
    assert result.provenance.loc[2, "Voltage.Vb"] == "last_observed"
    assert result.provenance.loc[0, "Voltage.Va"] != "lstm"


def test_default_warmup_policy_rejects_missing_values_without_context(waveform) -> None:
    config = ImputerConfig(
        sequence_length=4,
        hidden_size=8,
        num_layers=2,
        batch_size=16,
        max_epochs=1,
        patience=1,
        validation_fraction=0.2,
        test_fraction=0.2,
        max_gap=2,
        device="cpu",
    )
    imputer = LSTMImputer(config)
    imputer.fit([waveform])
    incomplete = waveform.copy()
    incomplete.loc[0, "Voltage.Va"] = np.nan

    with pytest.raises(InsufficientContextError, match="warmup_strategy"):
        imputer.impute(incomplete)


def test_post_warmup_imputation_provenance_is_lstm(waveform, tiny_config) -> None:
    imputer = LSTMImputer(tiny_config)
    imputer.fit([waveform])
    incomplete = waveform.copy()
    incomplete.loc[24, "Voltage.Va"] = np.nan

    result = imputer.impute(incomplete)

    assert result.provenance.loc[24, "Voltage.Va"] == "lstm"
