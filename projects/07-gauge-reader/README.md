# Gauge Reader

> Reading gauges automatically with edge AI.

**Status:** 🚧 In progress. Code is being added.

## Problem
_What is being read (analog dial, digital display…), where it is used, and why manual reading is a problem._

## Approach
_Pipeline from camera image to reading: detection / localisation, needle or digit recognition, conversion to a value. Target device._

## Results
| Metric | Value |
|---|---|
| Reading error (mean / max) | – |
| Model size | – |
| Inference latency on target | – |
| Target hardware | – |

## How to run
_Commands to train, evaluate and deploy._

## Layout
```
src/         Python: data preparation, training, evaluation, conversion
models/      exported models (small files only; large ones via release or download script)
firmware/    on-device code (e.g. ESP32 / embedded Linux)
notebooks/   experiments
data/        dataset description and download script (no large raw data in git)
results/     metrics, plots, sample predictions
docs/        diagrams and notes
```

## Adding code (conventions)
- Put reusable code in `src/`, not only in notebooks.
- Every result in the table above links to the script or notebook that produced it.
- Keep large files (datasets, model checkpoints) out of git.
