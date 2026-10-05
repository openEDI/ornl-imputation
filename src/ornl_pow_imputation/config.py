from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class ImputerConfig:
    target_columns: tuple[str, ...] = (
        "Voltage.Va",
        "Voltage.Vb",
        "Voltage.Vc",
    )
    timestamp_column: str = "Time µs"
    sequence_length: int = 32
    hidden_size: int = 80
    num_layers: int = 2
    dropout: float = 0.2
    learning_rate: float = 0.001
    batch_size: int = 128
    max_epochs: int = 100
    patience: int = 12
    validation_fraction: float = 0.15
    test_fraction: float = 0.15
    max_gap: int = 8
    seed: int = 42
    device: str = "auto"
    cadence_tolerance: float = 0.02
    warmup_strategy: str = "error"

    def __post_init__(self) -> None:
        self.target_columns = tuple(self.target_columns)
        if not self.target_columns or len(set(self.target_columns)) != len(self.target_columns):
            raise ValueError("target_columns must be non-empty and unique")
        if not self.timestamp_column:
            raise ValueError("timestamp_column must be non-empty")

        for name in (
            "sequence_length",
            "hidden_size",
            "num_layers",
            "batch_size",
            "max_epochs",
            "patience",
            "max_gap",
        ):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be greater than zero")

        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be greater than zero")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if not 0 <= self.validation_fraction < 1:
            raise ValueError("validation_fraction must be in [0, 1)")
        if not 0 <= self.test_fraction < 1:
            raise ValueError("test_fraction must be in [0, 1)")
        if self.validation_fraction + self.test_fraction >= 1:
            raise ValueError("validation and test fractions must leave training data")
        if not 0 < self.cadence_tolerance < 1:
            raise ValueError("cadence_tolerance must be in (0, 1)")
        if self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device must be one of: auto, cpu, cuda")
        if self.warmup_strategy not in {
            "error",
            "last_observed_then_training_median",
        }:
            raise ValueError("warmup_strategy must be error or last_observed_then_training_median")

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["target_columns"] = list(self.target_columns)
        return data
