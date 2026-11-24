# Lab 6: Training Pipelines and Profiling

Configuration: `lab06/config.yaml`; seed: 42; device: cpu.
Model: `CNN` imported from Lab 3; dataset: Fashion-MNIST.

## Evidence

- `profile_breakdown.csv` generated.
- `workers_study.csv` generated.
- `batch_study.csv` generated.
- `optimizer_study.csv` generated.
- `accumulation_study.csv` generated.
- `resume_demo.csv` generated.

## Interpretation
The fastest data-loading worker count depends on hardware and Windows process overhead.
Batch size changes throughput and memory use; GPU peak memory is unavailable on CPU.
Compare optimizer and scheduler results under equal ten-epoch budgets.
Gradient accumulation approximates a larger effective batch; exact equivalence can be affected by batch-dependent layers.
Resume demonstration uses the saved model, optimizer, scheduler, epoch and RNG states.

## Project proposal
Handwritten digit recognition on MNIST with a small CNN. Target: at least 97% test accuracy,
single-image inference <= 50 ms, model size <= 5 MB. Planned optimization: quantization.
These are targets that must be measured, not established results.
