# ORNL Point-on-Wave Data Imputation

This directory contains a standalone Python implementation of the point-on-wave
data-imputation method. It trains a Long Short-Term Memory (LSTM) model to fill missing values in three-phase voltage
streams.

## Install

Python 3.10 or newer.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

## Reference data

The default target columns are `Voltage.Va`, `Voltage.Vb`, and `Voltage.Vc`.
Training uses chronological splits, fits scaling parameters on training data
only, and saves the configuration, fitted scaler, model weights, and training
metrics together as a reusable artifact.
Each artifact is a directory containing `metadata.json` and `model.npz`; keep
both files together when moving or deploying a trained model.

The LSTM forecasts the scaled next-step residual (the change from the latest
sample) and adds it back to that sample. This formulation outperformed causal
forward-fill on both reference sampling interval groups. Sampling interval validation allows 2%
relative tolerance for normal source jitter while rejecting duplicates,
reversals, and roughly two-times cadence changes.

## Command line

Train a model artifact from one or more complete reference streams in the same
sampling interval group:

```powershell
ornl-pow-impute train `
  examples/reference/data/sigId-252.csv `
  examples/reference/data/sigId-253.csv `
  --artifact artifacts/pow-lstm-65024
```

Fill supported gaps in a CSV while preserving observed target values and all
non-target columns:

```powershell
ornl-pow-impute run `
  input-with-gaps.csv `
  --artifact artifacts/pow-lstm-65024 `
  --output completed.csv `
  --report completed.report.json
```

## Python
### Train, save, load, and batch-impute

```python
import pandas as pd

from ornl_pow_imputation import LSTMImputer

training_frames = [
    pd.read_csv("examples/reference/data/sigId-252.csv"),
    pd.read_csv("examples/reference/data/sigId-253.csv"),
]

imputer = LSTMImputer()
training_report = imputer.fit(training_frames)
imputer.save("artifacts/pow-lstm-65024")

restored = LSTMImputer.load("artifacts/pow-lstm-65024")
incomplete = pd.read_csv("input-with-gaps.csv")
result = restored.impute(incomplete)

result.data.to_csv("completed.csv", index=False)
print(result.imputed_mask)
print(result.gap_diagnostics)
```

Only missing cells in configured target columns are candidates for replacement.
Observed values and non-target columns are retained. Gaps longer than the
configured maximum are rejected instead of being silently extrapolated.
The default warm-up policy similarly rejects gaps when there is not enough
causal history. Applications may explicitly opt into
`last_observed_then_training_median`; every cell produced by that fallback is
identified in the result and JSON report.

### Stateful streaming

```python
from ornl_pow_imputation import LSTMImputer, StreamingLSTMImputer

model = LSTMImputer.load("artifacts/pow-lstm-65024")
stream = StreamingLSTMImputer(model)

result = stream.impute_sample(
    timestamp=1133827200000000000,
    values={"Voltage.Va": 120.1, "Voltage.Vb": None, "Voltage.Vc": -58.7},
)
print(result.values)
print(result.imputed_mask)
```

Keep one streaming imputer instance per ordered stream so its causal history is
not mixed across devices. The integration layer remains responsible for message
transport, ordering, partitioning, and OEDISI/HELICS lifecycle behavior.

Reference inputs form two sampling interval groups, labeled `130048` and
`65024` in the source data. Train and deploy a separate artifact for each
group; do not combine the groups in a single `train` invocation.

## Reference quality benchmark

The deterministic held-out missing-point benchmark reports aggregate RMSE:

| Sampling interval group | LSTM residual forecast | Causal forward-fill |
| --- | ---: | ---: |
| `130048` | 339.01 | 449.76 |
| `65024` | 125.34 | 379.16 |

Every voltage phase improved over forward-fill. Re-run the benchmark with:

```powershell
$env:RUN_REFERENCE_QUALITY=1; pytest -m reference_quality
```


