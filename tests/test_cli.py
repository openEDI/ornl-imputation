from __future__ import annotations

import json
import subprocess
import sys

import pandas as pd


def test_cli_train_and_run_smoke(tmp_path, waveform) -> None:
    training_csv = tmp_path / "training.csv"
    input_csv = tmp_path / "incomplete.csv"
    artifact = tmp_path / "artifact"
    output_csv = tmp_path / "imputed.csv"
    report_json = tmp_path / "report.json"
    waveform.to_csv(training_csv, index=False)
    incomplete = waveform.copy()
    incomplete.loc[20, "Voltage.Va"] = float("nan")
    incomplete.to_csv(input_csv, index=False)

    train = subprocess.run(
        [
            sys.executable,
            "-m",
            "ornl_pow_imputation.cli",
            "train",
            str(training_csv),
            "--artifact",
            str(artifact),
            "--max-epochs",
            "2",
            "--sequence-length",
            "4",
            "--hidden-size",
            "8",
            "--batch-size",
            "16",
            "--device",
            "cpu",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert train.returncode == 0, train.stderr

    run = subprocess.run(
        [
            sys.executable,
            "-m",
            "ornl_pow_imputation.cli",
            "run",
            str(input_csv),
            "--artifact",
            str(artifact),
            "--output",
            str(output_csv),
            "--report",
            str(report_json),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert run.returncode == 0, run.stderr
    completed = pd.read_csv(output_csv)
    assert pd.notna(completed.loc[20, "Voltage.Va"])
    assert completed.loc[0, "Voltage.Va"] == waveform.loc[0, "Voltage.Va"]
    report = json.loads(report_json.read_text(encoding="utf-8"))
    assert report["imputed_count"] == 1
    assert report["imputed_cells"] == [
        {
            "row": 20,
            "column": "Voltage.Va",
            "provenance": "lstm",
            "fallback_used": "",
        }
    ]
    assert report["target_columns"] == ["Voltage.Va", "Voltage.Vb", "Voltage.Vc"]
