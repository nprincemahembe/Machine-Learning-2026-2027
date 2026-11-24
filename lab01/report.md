# Lab 1: Environment and First System Measurements

## Goal
Compare Logistic Regression and Random Forest using the
Breast Cancer Wisconsin dataset.

## Method
The dataset was split into 70% training and 30% testing,
using stratification and random_state=42.

Training time was measured using one warm-up and five
timed runs. Inference time used one warm-up and 100
single-sample predictions.

Models were saved with joblib.

Memory is the Python process's cumulative peak working
set on Windows, not an isolated per-model peak.

## Results

| Model | Accuracy | Training ms | Inference ms | Size KB | Peak RSS MB |
|---|---:|---:|---:|---:|---:|
| logistic | 0.9415 | 398.236 | 0.1824 | 1.0303 | 104.395 |
| forest | 0.9357 | 277.9856 | 5.224 | 284.0713 | 105.473 |

## Deployment Comparison

| Model | Target | Fits measured upper limits? |
|---|---|---|
| logistic | Cloud | YES |
| logistic | Edge | YES |
| logistic | Mobile | YES |
| logistic | TinyML | NO |
| forest | Cloud | YES |
| forest | Edge | YES |
| forest | Mobile | YES |
| forest | TinyML | NO |

Cloud: 1 GB memory, 100 ms latency, 500 MB model.
Edge: 256-1024 MB memory, 50 ms latency, 50 MB model.
Mobile: 64-256 MB memory, 20 ms latency, 10 MB model.
TinyML: 256 KB memory, 10 ms latency, 100 KB model.

These are preliminary comparisons using Python process
memory. They do not confirm compatibility with actual
deployment hardware.

## Three Conclusions

1. [Write your accuracy comparison here.]
2. [Compare model speed and storage size here.]
3. [Explain deployment suitability and limitations here.]
