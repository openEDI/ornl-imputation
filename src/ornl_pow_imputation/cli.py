from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from .config import ImputerConfig
from .imputer import LSTMImputer


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ornl-pow-impute",
        description="Train and run the ORNL point-on-wave LSTM imputer.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    train = commands.add_parser("train", help="train and save a sampling-interval-specific model")
    train.add_argument("inputs", nargs="+", type=Path)
    train.add_argument("--artifact", type=Path, required=True)
    train.add_argument("--sequence-length", type=int, default=32)
    train.add_argument("--hidden-size", type=int, default=80)
    train.add_argument("--num-layers", type=int, default=2)
    train.add_argument("--dropout", type=float, default=0.2)
    train.add_argument("--learning-rate", type=float, default=0.001)
    train.add_argument("--batch-size", type=int, default=128)
    train.add_argument("--max-epochs", type=int, default=100)
    train.add_argument("--patience", type=int, default=12)
    train.add_argument("--max-gap", type=int, default=8)
    train.add_argument(
        "--warmup-strategy",
        choices=("error", "last_observed_then_training_median"),
        default="error",
    )
    train.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")

    run = commands.add_parser("run", help="impute missing voltage samples in a CSV")
    run.add_argument("input", type=Path)
    run.add_argument("--artifact", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--report", type=Path, required=True)
    run.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    return parser


def _expand_inputs(paths: Sequence[Path]) -> list[Path]:
    expanded: list[Path] = []
    for path in paths:
        if path.is_dir():
            expanded.extend(sorted(path.glob("*.csv")))
        else:
            expanded.append(path)
    if not expanded:
        raise ValueError("no CSV training inputs found")
    return expanded


def _train(args: argparse.Namespace) -> int:
    config = ImputerConfig(
        sequence_length=args.sequence_length,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        patience=args.patience,
        max_gap=args.max_gap,
        warmup_strategy=args.warmup_strategy,
        device=args.device,
    )
    paths = _expand_inputs(args.inputs)
    frames = [pd.read_csv(path) for path in paths]
    imputer = LSTMImputer(config)
    report = imputer.fit(frames)
    imputer.save(args.artifact)
    print(json.dumps({**report.__dict__, "artifact": str(args.artifact)}, indent=2))
    return 0


def _run(args: argparse.Namespace) -> int:
    imputer = LSTMImputer.load(args.artifact, device=args.device)
    frame = pd.read_csv(args.input)
    result = imputer.impute(frame)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    result.data.to_csv(args.output, index=False)
    provenance_counts = {
        value: int((result.provenance == value).to_numpy().sum())
        for value in ("observed", "lstm", "last_observed", "training_median")
    }
    imputed_cells = []
    for row in result.imputed_mask.index:
        row_value = row.item() if hasattr(row, "item") else row
        for column in imputer.config.target_columns:
            if result.imputed_mask.at[row, column]:
                imputed_cells.append(
                    {
                        "row": row_value,
                        "column": column,
                        "provenance": result.provenance.at[row, column],
                        "fallback_used": result.fallback_used.at[row, column],
                    }
                )
    report = {
        "imputed_count": int(result.imputed_mask.to_numpy().sum()),
        "imputed_cells": imputed_cells,
        "target_columns": list(imputer.config.target_columns),
        "max_gap_by_column": result.gap_diagnostics.max_gap_by_column,
        "provenance_counts": provenance_counts,
        "model_metadata": result.model_metadata,
    }
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    return _train(args) if args.command == "train" else _run(args)


if __name__ == "__main__":
    raise SystemExit(main())
