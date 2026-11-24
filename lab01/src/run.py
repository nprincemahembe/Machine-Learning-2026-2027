
import csv
import random
import statistics
import sys
import time
from pathlib import Path
from importlib.metadata import version, PackageNotFoundError

import joblib
import numpy as np
import psutil
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "lab01" / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

random.seed(42)
np.random.seed(42)


def save_csv(filename, rows):
    with (RESULTS / filename).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def benchmark(function, repeats):
    function()  
    times = []

    for _ in range(repeats):
        start = time.perf_counter()
        function()
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)

    return statistics.median(times)


def print_versions():
    packages = [
        "numpy", "pandas", "scikit-learn", "scipy",
        "matplotlib", "seaborn", "torch", "torchvision",
        "torchinfo", "thop", "onnx", "onnxruntime",
        "mlflow", "memory-profiler", "psutil",
        "codecarbon", "fastapi", "uvicorn", "pytest",
        "httpx", "locust", "requests", "pyarrow",
        "joblib", "tqdm"
    ]

    lines = [f"Python: {sys.version.split()[0]}"]

    for package in packages:
        try:
            installed = version(package)
        except PackageNotFoundError:
            installed = "NOT INSTALLED"

        lines.append(f"{package}: {installed}")

    output = "\n".join(lines)
    print(output)

    (RESULTS / "versions.txt").write_text(output)


def main():
    print_versions()

    X, y = load_breast_cancer(return_X_y=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.30,
        stratify=y,
        random_state=42
    )

    models = {
        "logistic": lambda: LogisticRegression(
            max_iter=1000,
            random_state=42
        ),
        "forest": lambda: RandomForestClassifier(
            n_estimators=100,
            random_state=42
        )
    }

    process = psutil.Process()
    rows = []

    for name, create_model in models.items():

        def train():
            model = create_model()
            model.fit(X_train, y_train)
            return model

        training_time = benchmark(train, 5)

        model = train()

        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)

        inference_time = benchmark(
            lambda: model.predict(X_test[:1]),
            100
        )

        model_file = RESULTS / f"{name}_model.joblib"
        joblib.dump(model, model_file)

        size_bytes = model_file.stat().st_size
        size_kb = size_bytes / 1024

        memory = process.memory_info()
        current_rss_mb = memory.rss / (1024 ** 2)

        # Windows cumulative process peak, not an
        # isolated peak for this model's operation.
        peak_rss_mb = (
            memory.peak_wset / (1024 ** 2)
            if hasattr(memory, "peak_wset")
            else current_rss_mb
        )

        rows.append({
            "model": name,
            "accuracy": f"{accuracy:.4f}",
            "training_median_ms": round(training_time, 4),
            "inference_median_ms": round(inference_time, 4),
            "size_bytes": size_bytes,
            "size_kb": round(size_kb, 4),
            "current_rss_mb": round(current_rss_mb, 3),
            "process_peak_rss_mb": round(peak_rss_mb, 3)
        })

        print(
            f"{name}: accuracy={accuracy:.4f}, "
            f"training={training_time:.4f} ms, "
            f"inference={inference_time:.4f} ms, "
            f"size={size_kb:.2f} KB"
        )

    save_csv("measurements.csv", rows)

    save_csv(
        "baseline_accuracy.csv",
        [
            {
                "model": row["model"],
                "test_accuracy": row["accuracy"]
            }
            for row in rows
        ]
    )

    budgets = {
        "Cloud": (1024, 100, 500 * 1024),
        "Edge": (1024, 50, 50 * 1024),
        "Mobile": (256, 20, 10 * 1024),
        "TinyML": (0.25, 10, 100)
    }

    deployment_rows = []

    for row in rows:
        for target, (memory_limit, latency_limit, size_limit) in budgets.items():
            fits = (
                row["process_peak_rss_mb"] <= memory_limit
                and row["inference_median_ms"] <= latency_limit
                and row["size_kb"] <= size_limit
            )

            deployment_rows.append({
                "model": row["model"],
                "deployment": target,
                "fits_measured_limits": "YES" if fits else "NO"
            })

    save_csv("deployment_fit.csv", deployment_rows)

    results_table = "\n".join(
        f"| {r['model']} | {r['accuracy']} | "
        f"{r['training_median_ms']} | "
        f"{r['inference_median_ms']} | "
        f"{r['size_kb']} | "
        f"{r['process_peak_rss_mb']} |"
        for r in rows
    )

    deployment_table = "\n".join(
        f"| {r['model']} | {r['deployment']} | "
        f"{r['fits_measured_limits']} |"
        for r in deployment_rows
    )

    report = f"""# Lab 1: Environment and First System Measurements

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
{results_table}

## Deployment Comparison

| Model | Target | Fits measured upper limits? |
|---|---|---|
{deployment_table}

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
"""

    (ROOT / "lab01" / "report.md").write_text(report)

    print("\nLab 1 completed.")
    print("Results saved in:", RESULTS)
    print("Report saved in:", ROOT / "lab01" / "report.md")


if __name__ == "__main__":
    main()
