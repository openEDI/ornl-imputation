from __future__ import annotations

import pytest

from ornl_pow_imputation import LSTMImputer, SchemaError, StreamingLSTMImputer


def test_fit_infers_and_records_training_cadence(waveform, tiny_config) -> None:
    imputer = LSTMImputer(tiny_config)

    report = imputer.fit([waveform])

    assert report.cadence == pytest.approx(1_000.0)
    assert imputer.training_cadence == pytest.approx(1_000.0)
    assert imputer.model_metadata["cadence"] == pytest.approx(1_000.0)
    assert imputer.model_metadata["timestamp_column"] == "Time µs"


def test_fit_rejects_frames_with_mixed_sampling_intervals(waveform, tiny_config) -> None:
    slower = waveform.copy()
    slower["Time µs"] *= 2

    with pytest.raises(SchemaError, match=r"cadence.*(1000|1,000).*(2000|2,000)"):
        LSTMImputer(tiny_config).fit([waveform, slower])


def test_impute_rejects_cadence_outside_tolerance(waveform, tiny_config) -> None:
    imputer = LSTMImputer(tiny_config)
    imputer.fit([waveform])
    mismatched = waveform.copy()
    mismatched["Time µs"] = mismatched.index.to_numpy() * 1_100

    with pytest.raises(SchemaError, match=r"cadence.*(1000|1,000).*(1100|1,100)"):
        imputer.impute(mismatched)


def test_loaded_artifact_retains_and_enforces_cadence(tmp_path, waveform, tiny_config) -> None:
    artifact = tmp_path / "pow-lstm"
    imputer = LSTMImputer(tiny_config)
    imputer.fit([waveform])
    imputer.save(artifact)
    restored = LSTMImputer.load(artifact, device="cpu")
    mismatched = waveform.copy()
    mismatched["Time µs"] = mismatched.index.to_numpy() * 1_100

    assert restored.training_cadence == pytest.approx(1_000.0)
    with pytest.raises(SchemaError, match="cadence"):
        restored.impute(mismatched)


def test_stream_rejects_timestamp_cadence_mismatch(waveform, tiny_config) -> None:
    imputer = LSTMImputer(tiny_config)
    imputer.fit([waveform])
    stream = StreamingLSTMImputer(imputer)
    values = waveform.iloc[0].drop(labels="Time µs").to_dict()
    stream.impute_sample(timestamp=0, values=values)

    with pytest.raises(SchemaError, match=r"cadence.*(1000|1,000).*(1100|1,100)"):
        stream.impute_sample(timestamp=1_100, values=values)


def test_stream_reports_warmup_fallback_provenance(waveform, tiny_config) -> None:
    imputer = LSTMImputer(tiny_config)
    imputer.fit([waveform])
    stream = StreamingLSTMImputer(imputer)
    values = waveform.iloc[0].drop(labels="Time µs").to_dict()
    values["Voltage.Va"] = float("nan")

    result = stream.impute_sample(timestamp=0, values=values)

    assert result.provenance["Voltage.Va"] == "training_median"
    assert result.provenance["Voltage.Va"] != "lstm"


@pytest.mark.parametrize("kind", ["dropped", "duplicate", "reversed", "nonfinite"])
def test_fit_rejects_each_invalid_timestamp_delta(waveform, tiny_config, kind) -> None:
    invalid = waveform.copy()
    if kind == "dropped":
        invalid.loc[20:, "Time µs"] += 1_000
    elif kind == "duplicate":
        invalid.loc[20, "Time µs"] = invalid.loc[19, "Time µs"]
    elif kind == "reversed":
        invalid.loc[20, "Time µs"] = invalid.loc[19, "Time µs"] - 1
    else:
        invalid["Time µs"] = invalid["Time µs"].astype(float)
        invalid.loc[20, "Time µs"] = float("nan")

    with pytest.raises(SchemaError, match=r"timestamp.*row 20"):
        LSTMImputer(tiny_config).fit([invalid])
