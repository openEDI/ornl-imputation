from __future__ import annotations

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

from ornl_pow_imputation import GapTooLongError, LSTMImputer, SchemaError, StreamingLSTMImputer


def test_batch_and_streaming_share_the_same_causal_results(waveform, tiny_config) -> None:
    trained = LSTMImputer(tiny_config)
    trained.fit([waveform])
    incomplete = waveform.copy()
    incomplete.loc[20, "Voltage.Va"] = np.nan
    incomplete.loc[36:37, "Voltage.Vc"] = np.nan
    batch = trained.impute(incomplete)
    stream = StreamingLSTMImputer(trained)

    streamed_rows: list[dict[str, object]] = []
    streamed_masks: list[dict[str, bool]] = []
    for _, row in incomplete.iterrows():
        result = stream.impute_sample(
            timestamp=row["Time µs"],
            values=row.drop(labels="Time µs").to_dict(),
        )
        streamed_rows.append({"Time µs": result.timestamp, **result.values})
        streamed_masks.append(result.imputed_mask)

    streamed = pd.DataFrame(streamed_rows, columns=incomplete.columns)
    stream_mask = pd.DataFrame(streamed_masks)[list(tiny_config.target_columns)]
    pdt.assert_frame_equal(streamed, batch.data, check_exact=True)
    pdt.assert_frame_equal(
        stream_mask.reset_index(drop=True),
        batch.imputed_mask.reset_index(drop=True),
    )


def test_streaming_preserves_observed_values(waveform, tiny_config) -> None:
    trained = LSTMImputer(tiny_config)
    trained.fit([waveform])
    stream = StreamingLSTMImputer(trained)
    row = waveform.iloc[0]

    result = stream.impute_sample(
        timestamp=row["Time µs"], values=row.drop(labels="Time µs").to_dict()
    )

    for column in tiny_config.target_columns:
        assert result.values[column] == np.float32(row[column])
        assert result.imputed_mask[column] is False


def test_rejected_stream_sample_does_not_advance_timestamp_or_history(
    waveform, tiny_config
) -> None:
    trained = LSTMImputer(tiny_config)
    trained.fit([waveform])
    stream = StreamingLSTMImputer(trained)
    first = waveform.iloc[0]
    stream.impute_sample(timestamp=first["Time µs"], values=first.drop(labels="Time µs").to_dict())
    second = waveform.iloc[1]
    invalid = second.drop(labels="Time µs").to_dict()
    invalid["Voltage.Va"] = float("inf")

    with pytest.raises(SchemaError, match="non-finite"):
        stream.impute_sample(timestamp=second["Time µs"], values=invalid)

    retried = stream.impute_sample(
        timestamp=second["Time µs"], values=second.drop(labels="Time µs").to_dict()
    )
    assert retried.timestamp == second["Time µs"]


def test_rejected_overlong_gap_can_be_retried_and_reset(waveform, tiny_config) -> None:
    trained = LSTMImputer(tiny_config)
    trained.fit([waveform])
    stream = StreamingLSTMImputer(trained)
    for position in range(tiny_config.sequence_length):
        row = waveform.iloc[position]
        stream.impute_sample(timestamp=row["Time µs"], values=row.drop(labels="Time µs").to_dict())

    for position in range(tiny_config.sequence_length, tiny_config.sequence_length + 2):
        row = waveform.iloc[position]
        values = row.drop(labels="Time µs").to_dict()
        values["Voltage.Va"] = float("nan")
        stream.impute_sample(timestamp=row["Time µs"], values=values)

    rejected = waveform.iloc[tiny_config.sequence_length + 2]
    missing = rejected.drop(labels="Time µs").to_dict()
    missing["Voltage.Va"] = float("nan")
    with pytest.raises(GapTooLongError):
        stream.impute_sample(timestamp=rejected["Time µs"], values=missing)

    retried = stream.impute_sample(
        timestamp=rejected["Time µs"], values=rejected.drop(labels="Time µs").to_dict()
    )
    assert retried.imputed_mask["Voltage.Va"] is False

    stream.reset()
    first = waveform.iloc[0]
    replayed = stream.impute_sample(
        timestamp=first["Time µs"], values=first.drop(labels="Time µs").to_dict()
    )
    assert replayed.timestamp == first["Time µs"]
