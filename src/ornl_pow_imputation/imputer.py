from __future__ import annotations

import json
import math
import os
import platform
import random
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn
import torch
from sklearn.preprocessing import RobustScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ._version import __version__
from .config import ImputerConfig
from .model import LSTMRegressor, build_sequences
from .results import (
    GapDiagnostics,
    ImputationResult,
    StreamResult,
    TrainingReport,
)


class ImputationError(Exception):
    pass


class SchemaError(ImputationError):
    """Raised when input data does not satisfy the configured data contract."""


class GapTooLongError(ImputationError):
    """Raised when a missing run exceeds the validated causal gap limit."""


class InsufficientContextError(ImputationError):
    """Raised when causal inference has not accumulated enough prior samples."""


class ArtifactError(ImputationError):
    """Raised when a saved artifact uses an unsupported schema."""


ARTIFACT_SCHEMA_VERSION = 1


def _format_number(value: float) -> str:
    return f"{value:g}"


def _timestamp_values(series: pd.Series) -> np.ndarray:
    if pd.api.types.is_datetime64_any_dtype(series.dtype):
        return series.astype("int64").to_numpy()
    if not pd.api.types.is_numeric_dtype(series.dtype):
        try:
            converted = pd.to_datetime(series, errors="raise")
        except (TypeError, ValueError) as exc:
            raise SchemaError(
                f"timestamp column {series.name!r} must be numeric or datetime"
            ) from exc
        return converted.astype("int64").to_numpy()
    values = series.to_numpy()
    finite = np.isfinite(values.astype(np.float64))
    if not np.all(finite):
        position = int(np.flatnonzero(~finite)[0])
        raise SchemaError(
            f"timestamp column {series.name!r} has a non-finite value at row "
            f"{series.index[position]!r}"
        )
    return values


def _infer_cadence(frame: pd.DataFrame, timestamp_column: str, tolerance: float) -> float | None:
    if timestamp_column not in frame.columns or len(frame) < 2:
        return None
    values = _timestamp_values(frame[timestamp_column])
    differences = np.diff(values)
    if np.any(differences <= 0):
        position = int(np.flatnonzero(differences <= 0)[0]) + 1
        raise SchemaError(
            f"timestamp column {timestamp_column!r} must be strictly increasing; "
            f"invalid value at row {frame.index[position]!r}"
        )
    cadence = float(np.median(differences))
    allowed = max(abs(cadence) * tolerance, np.finfo(float).eps)
    irregular = np.abs(differences.astype(np.float64) - cadence) > allowed
    if np.any(irregular):
        position = int(np.flatnonzero(irregular)[0]) + 1
        raise SchemaError(
            f"timestamp column {timestamp_column!r} has irregular cadence around "
            f"{_format_number(cadence)} at row {frame.index[position]!r}"
        )
    return cadence


def _cadence_matches(expected: float, actual: float, tolerance: float) -> bool:
    return math.isclose(expected, actual, rel_tol=tolerance, abs_tol=0.0)


class LSTMImputer:
    """Train, persist, and apply a cadence-aware causal LSTM imputer."""

    def __init__(self, config: ImputerConfig | None = None) -> None:
        self.config = config or ImputerConfig()
        self.device = self._resolve_device(self.config.device)
        self.model: LSTMRegressor | None = None
        self.scaler: RobustScaler | None = None
        self.training_medians: dict[str, float] = {}
        self.training_cadence: float | None = None
        self.training_timestamp_dtype: str | None = None
        self.training_dtypes: dict[str, str] = {}
        self.model_metadata: dict[str, Any] = {}
        self.training_report: TrainingReport | None = None

    @staticmethod
    def _resolve_device(requested: str) -> torch.device:
        if requested == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if requested == "cuda" and not torch.cuda.is_available():
            raise ValueError("device='cuda' requested but CUDA is unavailable")
        return torch.device(requested)

    def _validate_frame(self, frame: pd.DataFrame, *, require_complete: bool) -> None:
        if not isinstance(frame, pd.DataFrame):
            raise SchemaError("input must be a pandas DataFrame")
        for column in self.config.target_columns:
            if column not in frame.columns:
                raise SchemaError(f"missing required target column {column!r}")
            if not pd.api.types.is_numeric_dtype(frame[column].dtype):
                raise SchemaError(f"target column {column!r} must be numeric")
            values = frame[column].to_numpy(dtype=np.float64, na_value=np.nan)
            invalid = ~np.isfinite(values) if require_complete else np.isinf(values)
            if np.any(invalid):
                position = int(np.flatnonzero(invalid)[0])
                qualifier = "training " if require_complete else ""
                raise SchemaError(
                    f"{qualifier}target column {column!r} has a non-finite value "
                    f"at row {frame.index[position]!r}"
                )
        if require_complete and len(frame) <= self.config.sequence_length:
            raise SchemaError(f"training frame needs more than {self.config.sequence_length} rows")

    def _check_training_cadences(self, frames: Sequence[pd.DataFrame]) -> float | None:
        cadences = [
            _infer_cadence(
                frame,
                self.config.timestamp_column,
                self.config.cadence_tolerance,
            )
            for frame in frames
        ]
        observed = [cadence for cadence in cadences if cadence is not None]
        if not observed:
            return None
        expected = observed[0]
        for actual in observed[1:]:
            if not _cadence_matches(expected, actual, self.config.cadence_tolerance):
                raise SchemaError(
                    "cadence mismatch between training frames: "
                    f"{_format_number(expected)} and {_format_number(actual)}"
                )
        if len(observed) != len(frames):
            raise SchemaError(
                f"all training frames must include timestamp column "
                f"{self.config.timestamp_column!r} when cadence is available"
            )
        return expected

    def _check_inference_cadence(self, frame: pd.DataFrame) -> None:
        if self.training_cadence is None:
            return
        actual = _infer_cadence(
            frame,
            self.config.timestamp_column,
            self.config.cadence_tolerance,
        )
        if actual is None:
            raise SchemaError(
                f"input must include timestamp column {self.config.timestamp_column!r}"
            )
        if not _cadence_matches(self.training_cadence, actual, self.config.cadence_tolerance):
            raise SchemaError(
                "cadence mismatch: trained on "
                f"{_format_number(self.training_cadence)}, received "
                f"{_format_number(actual)}"
            )

    def _new_model(self) -> LSTMRegressor:
        return LSTMRegressor(
            input_size=len(self.config.target_columns),
            hidden_size=self.config.hidden_size,
            output_size=len(self.config.target_columns),
            num_layers=self.config.num_layers,
            dropout=self.config.dropout,
        ).to(self.device)

    def fit(self, frames: Sequence[pd.DataFrame]) -> TrainingReport:
        if not frames:
            raise SchemaError("at least one training frame is required")
        materialized = [frame.copy() for frame in frames]
        for frame in materialized:
            self._validate_frame(frame, require_complete=True)
        self.training_cadence = self._check_training_cadences(materialized)
        timestamp_dtypes = {
            str(frame[self.config.timestamp_column].dtype)
            for frame in materialized
            if self.config.timestamp_column in frame.columns
        }
        if len(timestamp_dtypes) > 1:
            raise SchemaError("training timestamp columns must use the same dtype")
        self.training_timestamp_dtype = next(iter(timestamp_dtypes), None)
        self.training_dtypes = {
            column: str(dtype) for column, dtype in materialized[0].dtypes.items()
        }

        train_parts: list[np.ndarray] = []
        validation_parts: list[np.ndarray] = []
        test_parts: list[np.ndarray] = []
        for frame in materialized:
            values = frame.loc[:, self.config.target_columns].to_numpy(dtype=np.float64)
            train_end = int(
                len(values) * (1 - self.config.validation_fraction - self.config.test_fraction)
            )
            validation_end = int(len(values) * (1 - self.config.test_fraction))
            if train_end <= self.config.sequence_length:
                raise SchemaError("training split is too short for the configured sequence_length")
            train_parts.append(values[:train_end])
            validation_parts.append(values[train_end:validation_end])
            test_parts.append(values[validation_end:])

        training_values = np.concatenate(train_parts, axis=0)
        self.scaler = RobustScaler().fit(training_values)
        medians = np.median(training_values, axis=0)
        self.training_medians = dict(
            zip(self.config.target_columns, medians.astype(float), strict=True)
        )

        train_x, train_y = self._sequence_parts(train_parts)
        validation_x, validation_y = self._sequence_parts(validation_parts)
        test_x, test_y = self._sequence_parts(test_parts)
        if len(train_x) == 0:
            raise SchemaError("no training sequences were produced")

        random.seed(self.config.seed)
        np.random.seed(self.config.seed)
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        torch.manual_seed(self.config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config.seed)
        self.model = self._new_model()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        criterion = nn.MSELoss()
        generator = torch.Generator().manual_seed(self.config.seed)
        loader = DataLoader(
            TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y)),
            batch_size=self.config.batch_size,
            shuffle=True,
            generator=generator,
        )

        best_loss = math.inf
        best_training_loss = math.inf
        best_state: dict[str, torch.Tensor] | None = None
        stale_epochs = 0
        epochs_trained = 0
        for epoch in range(self.config.max_epochs):
            self.model.train()
            loss_total = 0.0
            sample_total = 0
            for features, targets in loader:
                features = features.to(self.device)
                targets = targets.to(self.device)
                optimizer.zero_grad()
                predictions = self.model(features)
                loss = criterion(predictions, targets)
                loss.backward()
                optimizer.step()
                loss_total += float(loss.detach().cpu()) * len(features)
                sample_total += len(features)
            training_loss = loss_total / sample_total
            validation_loss = self._loss(validation_x, validation_y, criterion)
            score = validation_loss if math.isfinite(validation_loss) else training_loss
            epochs_trained = epoch + 1
            if score < best_loss - 1e-12:
                best_loss = score
                best_training_loss = training_loss
                best_state = {
                    name: value.detach().cpu().clone()
                    for name, value in self.model.state_dict().items()
                }
                stale_epochs = 0
            else:
                stale_epochs += 1
                if stale_epochs >= self.config.patience:
                    break

        if best_state is None:
            raise RuntimeError("training failed to produce a model state")
        self.model.load_state_dict(best_state)
        test_loss = self._loss(test_x, test_y, criterion)
        reported_validation_loss = float(best_loss) if len(validation_x) else None
        reported_test_loss = float(test_loss) if math.isfinite(test_loss) else None
        self.training_report = TrainingReport(
            training_loss=float(best_training_loss),
            validation_loss=reported_validation_loss,
            test_loss=reported_test_loss,
            epochs_trained=epochs_trained,
            training_sequences=len(train_x),
            cadence=self.training_cadence,
        )
        self.model_metadata = {
            "algorithm": "causal_lstm_residual_forecast",
            "target_columns": list(self.config.target_columns),
            "timestamp_column": self.config.timestamp_column,
            "timestamp_dtype": self.training_timestamp_dtype,
            "training_dtypes": self.training_dtypes,
            "cadence": self.training_cadence,
            "cadence_tolerance": self.config.cadence_tolerance,
            "warmup_strategy": self.config.warmup_strategy,
            "deterministic_algorithms": True,
            "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
            "runtime_versions": {
                "package": __version__,
                "python": platform.python_version(),
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "torch": torch.__version__,
                "scikit_learn": sklearn.__version__,
            },
            "config": self.config.to_dict(),
        }
        return self.training_report

    def _sequence_parts(self, parts: Sequence[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        if self.scaler is None:
            raise RuntimeError("scaler is not fitted")
        features: list[np.ndarray] = []
        targets: list[np.ndarray] = []
        for part in parts:
            if len(part) <= self.config.sequence_length:
                continue
            scaled = self.scaler.transform(part).astype(np.float32)
            part_x, part_y = build_sequences(scaled, self.config.sequence_length)
            part_y = part_y - part_x[:, -1, :]
            features.append(part_x)
            targets.append(part_y)
        if not features:
            width = len(self.config.target_columns)
            return (
                np.empty((0, self.config.sequence_length, width), dtype=np.float32),
                np.empty((0, width), dtype=np.float32),
            )
        return np.concatenate(features), np.concatenate(targets)

    def _loss(self, features: np.ndarray, targets: np.ndarray, criterion: nn.Module) -> float:
        if len(features) == 0:
            return float("nan")
        if self.model is None:
            raise RuntimeError("model is not fitted")
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(torch.from_numpy(features).to(self.device))
            target_tensor = torch.from_numpy(targets).to(self.device)
            return float(criterion(predictions, target_tensor).detach().cpu())

    def _require_fitted(self) -> None:
        if self.model is None or self.scaler is None or not self.training_medians:
            raise RuntimeError("imputer must be fitted or loaded before use")

    def _predict_next(self, history: Sequence[np.ndarray]) -> np.ndarray:
        self._require_fitted()
        if len(history) < self.config.sequence_length:
            raise RuntimeError("insufficient history for LSTM inference")
        assert self.scaler is not None
        assert self.model is not None
        window = np.stack(list(history)[-self.config.sequence_length :])
        scaled = self.scaler.transform(window).astype(np.float32)
        tensor = torch.from_numpy(scaled).unsqueeze(0).to(self.device)
        self.model.eval()
        with torch.no_grad():
            scaled_prediction = self.model(tensor).detach().cpu().numpy()
        scaled_prediction = scaled_prediction + scaled[-1:, :]
        return self.scaler.inverse_transform(scaled_prediction)[0]

    def impute(self, frame: pd.DataFrame) -> ImputationResult:
        self._require_fitted()
        self._validate_frame(frame, require_complete=False)
        self._check_inference_cadence(frame)
        diagnostics = self._gap_diagnostics(frame)
        for column, length in diagnostics.max_gap_by_column.items():
            if length > self.config.max_gap:
                raise GapTooLongError(
                    f"{column} has gap length {length}, exceeding max_gap {self.config.max_gap}"
                )

        completed = frame.copy(deep=True)
        imputed_mask = frame.loc[:, self.config.target_columns].isna()
        fallback_used = pd.DataFrame("", index=frame.index, columns=self.config.target_columns)
        provenance = pd.DataFrame("observed", index=frame.index, columns=self.config.target_columns)
        stream = StreamingLSTMImputer(self)
        for position, index in enumerate(frame.index):
            timestamp = (
                frame.iloc[position][self.config.timestamp_column]
                if self.config.timestamp_column in frame.columns
                else position
            )
            values = (
                frame.iloc[position]
                .drop(labels=[self.config.timestamp_column], errors="ignore")
                .to_dict()
            )
            result = stream.impute_sample(timestamp=timestamp, values=values)
            for column in self.config.target_columns:
                if result.imputed_mask[column]:
                    completed.at[index, column] = result.values[column]
                fallback_used.at[index, column] = result.fallback_used[column]
                provenance.at[index, column] = result.provenance[column]

        return ImputationResult(
            data=completed,
            imputed_mask=imputed_mask,
            fallback_used=fallback_used,
            provenance=provenance,
            gap_diagnostics=diagnostics,
            model_metadata=dict(self.model_metadata),
        )

    def _gap_diagnostics(self, frame: pd.DataFrame) -> GapDiagnostics:
        maximums: dict[str, int] = {}
        for column in self.config.target_columns:
            current = 0
            maximum = 0
            for missing in frame[column].isna().to_numpy():
                current = current + 1 if missing else 0
                maximum = max(maximum, current)
            maximums[column] = maximum
        return GapDiagnostics(max_gap_by_column=maximums)

    def save(self, artifact: str | Path) -> None:
        self._require_fitted()
        destination = Path(artifact)
        destination.mkdir(parents=True, exist_ok=True)
        assert self.model is not None
        assert self.scaler is not None
        state = {
            name: value.detach().cpu().numpy() for name, value in self.model.state_dict().items()
        }
        np.savez_compressed(destination / "model.npz", **state)
        payload = {
            "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
            "model_metadata": self.model_metadata,
            "training_medians": self.training_medians,
            "training_cadence": self.training_cadence,
            "training_timestamp_dtype": self.training_timestamp_dtype,
            "training_dtypes": self.training_dtypes,
            "scaler_center": self.scaler.center_.tolist(),
            "scaler_scale": self.scaler.scale_.tolist(),
            "training_report": asdict(self.training_report) if self.training_report else None,
        }
        (destination / "metadata.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, artifact: str | Path, device: str = "cpu") -> LSTMImputer:
        source = Path(artifact)
        payload = json.loads((source / "metadata.json").read_text(encoding="utf-8"))
        schema_version = payload.get("artifact_schema_version")
        if schema_version != ARTIFACT_SCHEMA_VERSION:
            raise ArtifactError(
                f"unsupported artifact schema version {schema_version!r}; "
                f"expected {ARTIFACT_SCHEMA_VERSION}"
            )
        config_data = dict(payload["model_metadata"]["config"])
        config_data["device"] = device
        imputer = cls(ImputerConfig(**config_data))
        imputer.model = imputer._new_model()
        with np.load(source / "model.npz", allow_pickle=False) as weights:
            state = {
                name: torch.from_numpy(weights[name]).to(imputer.device) for name in weights.files
            }
        imputer.model.load_state_dict(state)
        imputer.model.eval()
        imputer.scaler = RobustScaler()
        imputer.scaler.center_ = np.asarray(payload["scaler_center"], dtype=np.float64)
        imputer.scaler.scale_ = np.asarray(payload["scaler_scale"], dtype=np.float64)
        imputer.scaler.n_features_in_ = len(imputer.config.target_columns)
        imputer.training_medians = {
            key: float(value) for key, value in payload["training_medians"].items()
        }
        imputer.training_cadence = payload["training_cadence"]
        imputer.training_timestamp_dtype = payload.get("training_timestamp_dtype")
        imputer.training_dtypes = payload.get("training_dtypes", {})
        imputer.model_metadata = payload["model_metadata"]
        report = payload.get("training_report")
        imputer.training_report = TrainingReport(**report) if report else None
        return imputer


class StreamingLSTMImputer:
    """Stateful, causal sample-by-sample interface over a fitted imputer."""

    def __init__(self, imputer: LSTMImputer) -> None:
        imputer._require_fitted()
        self.imputer = imputer
        self.reset()

    def reset(self) -> None:
        self._history: deque[np.ndarray] = deque(maxlen=self.imputer.config.sequence_length)
        self._last_timestamp: Any = None
        self._gap_lengths = {column: 0 for column in self.imputer.config.target_columns}

    def _validate_timestamp(self, timestamp: Any) -> None:
        expected = self.imputer.training_cadence
        if self._last_timestamp is None:
            return
        if expected is not None:
            previous = pd.Series([self._last_timestamp, timestamp], name="timestamp")
            values = _timestamp_values(previous)
            actual = float(values[1] - values[0])
            if actual <= 0 or not _cadence_matches(
                expected, actual, self.imputer.config.cadence_tolerance
            ):
                raise SchemaError(
                    "cadence mismatch: trained on "
                    f"{_format_number(expected)}, received {_format_number(actual)}"
                )

    def _normalize_timestamp(self, timestamp: Any) -> Any:
        dtype = self.imputer.training_timestamp_dtype or ""
        if dtype.startswith(("int", "uint")) and isinstance(timestamp, (int, float, np.number)):
            numeric = float(timestamp)
            if not math.isfinite(numeric) or not numeric.is_integer():
                raise SchemaError(
                    f"timestamp {timestamp!r} is incompatible with trained dtype {dtype}"
                )
            return int(numeric)
        return timestamp

    def impute_sample(self, timestamp: Any, values: Mapping[str, Any]) -> StreamResult:
        timestamp = self._normalize_timestamp(timestamp)
        self._validate_timestamp(timestamp)
        output = dict(values)
        missing: dict[str, bool] = {}
        fallback: dict[str, str] = {}
        provenance: dict[str, str] = {}
        next_gap_lengths = dict(self._gap_lengths)
        for column in self.imputer.config.target_columns:
            if column not in output:
                raise SchemaError(f"missing required target column {column!r}")
            value = output[column]
            try:
                is_missing = bool(pd.isna(value))
            except (TypeError, ValueError) as exc:
                raise SchemaError(f"invalid value for target column {column!r}") from exc
            if not is_missing and not isinstance(value, (int, float, np.number)):
                raise SchemaError(f"target column {column!r} must be numeric")
            if not is_missing and not math.isfinite(float(value)):
                raise SchemaError(f"target column {column!r} has a non-finite value")
            missing[column] = is_missing
            next_gap_lengths[column] = next_gap_lengths[column] + 1 if is_missing else 0
            if next_gap_lengths[column] > self.imputer.config.max_gap:
                raise GapTooLongError(
                    f"{column} has gap length {next_gap_lengths[column]}, "
                    f"exceeding max_gap {self.imputer.config.max_gap}"
                )

        prediction: np.ndarray | None = None
        if any(missing.values()) and len(self._history) >= self.imputer.config.sequence_length:
            prediction = self.imputer._predict_next(self._history)

        for position, column in enumerate(self.imputer.config.target_columns):
            if not missing[column]:
                fallback[column] = ""
                provenance[column] = "observed"
                continue
            if prediction is not None:
                output[column] = float(prediction[position])
                fallback[column] = ""
                provenance[column] = "lstm"
            elif self.imputer.config.warmup_strategy == "error":
                raise InsufficientContextError(
                    f"missing {column!r} before {self.imputer.config.sequence_length} "
                    "prior samples were available; set warmup_strategy="
                    "'last_observed_then_training_median' to enable explicit fallback"
                )
            elif self._history:
                output[column] = float(self._history[-1][position])
                fallback[column] = "last_observed"
                provenance[column] = "last_observed"
            else:
                output[column] = self.imputer.training_medians[column]
                fallback[column] = "training_median"
                provenance[column] = "training_median"

        completed = np.asarray(
            [output[column] for column in self.imputer.config.target_columns],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(completed)):
            raise ImputationError("imputation produced a non-finite target value")
        for column, value in tuple(output.items()):
            dtype_name = self.imputer.training_dtypes.get(column)
            if dtype_name and not pd.isna(value):
                dtype = np.dtype(dtype_name)
                if np.issubdtype(dtype, np.number):
                    output[column] = dtype.type(value)
        self._last_timestamp = timestamp
        self._gap_lengths = next_gap_lengths
        self._history.append(completed)
        return StreamResult(
            timestamp=timestamp,
            values=output,
            imputed_mask=missing,
            fallback_used=fallback,
            provenance=provenance,
        )
