from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class TrainingReport:
    training_loss: float
    validation_loss: float | None
    test_loss: float | None
    epochs_trained: int
    training_sequences: int
    cadence: float | None


@dataclass(frozen=True)
class GapDiagnostics:
    max_gap_by_column: dict[str, int]


@dataclass(frozen=True)
class ImputationResult:
    data: pd.DataFrame
    imputed_mask: pd.DataFrame
    fallback_used: pd.DataFrame
    provenance: pd.DataFrame
    gap_diagnostics: GapDiagnostics
    model_metadata: dict[str, Any]


@dataclass(frozen=True)
class StreamResult:
    timestamp: Any
    values: dict[str, Any]
    imputed_mask: dict[str, bool]
    fallback_used: dict[str, str]
    provenance: dict[str, str]
