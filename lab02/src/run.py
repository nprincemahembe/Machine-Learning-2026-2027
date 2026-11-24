
import sys
import csv
import time
import random
import statistics
from pathlib import Path
from importlib.metadata import version, PackageNotFoundError

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "lab02" / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

random.seed(42)
np.random.seed(42)

print("Starting Machine Learning Lab 2...")


def save_versions():
    packages = [
        "numpy", "pandas", "scikit-learn",
        "scipy", "matplotlib", "seaborn",
        "torch", "torchvision", "torchinfo",
        "thop", "onnx", "onnxruntime",
        "mlflow", "memory-profiler", "psutil",
        "codecarbon", "fastapi", "uvicorn",
        "pytest", "httpx", "locust",
        "requests", "pyarrow", "joblib",
        "tqdm"
    ]

    lines = [f"Python: {sys.version.split()[0]}"]

    for package in packages:
        try:
            installed = version(package)
        except PackageNotFoundError:
            installed = "NOT INSTALLED"

        lines.append(f"{package}: {installed}")

    output = "\n".join(lines)
    (RESULTS / "versions.txt").write_text(output)
    print(output)


save_versions()


X, y = load_digits(return_X_y=True)


X = X.astype(np.float64) / 16.0

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    stratify=y,
    random_state=42
)


Y_train = np.eye(10)[y_train]

print("\nDataset loaded successfully.")
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))
print("Input features:", X.shape[1])


class NeuralNetwork:

    def __init__(self, width):

        self.width = width

        

        self.W1 = np.random.normal(
            0, 0.01, (64, width)
        )
        self.b1 = np.zeros(width)

        self.W2 = np.random.normal(
            0, 0.01, (width, width)
        )
        self.b2 = np.zeros(width)

        self.W3 = np.random.normal(
            0, 0.01, (width, 10)
        )
        self.b3 = np.zeros(10)

    def relu(self, x):
        return np.maximum(0, x)

    def softmax(self, x):
        shifted = x - np.max(
            x, axis=1, keepdims=True
        )
        exp_values = np.exp(shifted)

        return exp_values / np.sum(
            exp_values, axis=1, keepdims=True
        )


    def forward(self, X):

        z1 = X @ self.W1 + self.b1
        a1 = self.relu(z1)

        z2 = a1 @ self.W2 + self.b2
        a2 = self.relu(z2)

        z3 = a2 @ self.W3 + self.b3
        probabilities = self.softmax(z3)

        cache = (X, z1, a1, z2, a2, probabilities)

        return probabilities, cache

  

    def loss(self, probabilities, Y):

        probabilities = np.clip(
            probabilities, 1e-12, 1.0
        )

        return -np.mean(
            np.sum(
                Y * np.log(probabilities),
                axis=1
            )
        )

   
    def backward(self, cache, Y):

        X, z1, a1, z2, a2, probabilities = cache

        batch_size = len(X)

        dz3 = (probabilities - Y) / batch_size

        dW3 = a2.T @ dz3
        db3 = np.sum(dz3, axis=0)

        da2 = dz3 @ self.W3.T
        dz2 = da2 * (z2 > 0)

        dW2 = a1.T @ dz2
        db2 = np.sum(dz2, axis=0)

        da1 = dz2 @ self.W2.T
        dz1 = da1 * (z1 > 0)

        dW1 = X.T @ dz1
        db1 = np.sum(dz1, axis=0)

        return [
            dW1, db1,
            dW2, db2,
            dW3, db3
        ]

    

    def update(self, gradients, learning_rate):

        dW1, db1, dW2, db2, dW3, db3 = gradients

        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1

        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2

        self.W3 -= learning_rate * dW3
        self.b3 -= learning_rate * db3


    def predict(self, X):

        probabilities, _ = self.forward(X)

        return np.argmax(probabilities, axis=1)


np.random.seed(42)
check_net = NeuralNetwork(16)

probabilities, _ = check_net.forward(
    X_train[:32]
)

assert probabilities.shape == (32, 10)

assert np.allclose(
    probabilities.sum(axis=1),
    1.0,
    atol=1e-6
)

initial_loss = check_net.loss(
    probabilities,
    Y_train[:32]
)

print("\nForward pass check: PASSED")
print("Initial loss:", round(initial_loss, 6))
print("Expected loss: approximately 2.3026")

assert abs(initial_loss - np.log(10)) < 0.1


def gradient_check(network, X, Y):

    epsilon = 1e-5

    probabilities, cache = network.forward(X)
    analytical_gradients = network.backward(cache, Y)

    parameters = [
        network.W1, network.b1,
        network.W2, network.b2,
        network.W3, network.b3
    ]

    rng = np.random.default_rng(42)
    checks = []

    # Choose random weights with measurable gradients.
    # Very small gradients are sensitive to numerical rounding.
    candidates = []

    for matrix_index in [0, 2, 4]:
        gradient = analytical_gradients[matrix_index]

        for index in np.ndindex(gradient.shape):
            if abs(gradient[index]) >= 1e-5:
                candidates.append((matrix_index, index))

    if len(candidates) < 10:
        raise RuntimeError(
            "Not enough measurable gradients for checking."
        )

    selected = rng.choice(
        len(candidates),
        size=10,
        replace=False
    )

    for i, candidate_index in enumerate(selected):

        matrix_index, index = candidates[int(candidate_index)]

        parameter = parameters[matrix_index]
        analytical = analytical_gradients[matrix_index]

        original = parameter[index]

        try:
            parameter[index] = original + epsilon
            plus_probabilities, _ = network.forward(X)
            loss_plus = network.loss(plus_probabilities, Y)

            parameter[index] = original - epsilon
            minus_probabilities, _ = network.forward(X)
            loss_minus = network.loss(minus_probabilities, Y)

        finally:
            parameter[index] = original

        numerical = (loss_plus - loss_minus) / (2 * epsilon)
        analytic_value = analytical[index]

        relative_error = (
            abs(numerical - analytic_value)
            / max(
                1e-12,
                abs(numerical) + abs(analytic_value)
            )
        )

        checks.append({
            "check": i + 1,
            "parameter": f"W{matrix_index // 2 + 1}",
            "index": str(index),
            "numerical": numerical,
            "analytical": analytic_value,
            "relative_error": relative_error,
            "passed": relative_error < 1e-6
        })

    return checks



print("\nRunning numerical gradient check...")

checks = gradient_check(
    check_net,
    X_train[:16],
    Y_train[:16]
)

passed = sum(row["passed"] for row in checks)

for row in checks:
    print(
        f"Check {row['check']}: "
        f"error={row['relative_error']:.2e}, "
        f"passed={row['passed']}"
    )

print(f"Gradient check: {passed}/10 passed")

with (RESULTS / "gradient_check.csv").open(
    "w", newline=""
) as f:

    writer = csv.DictWriter(
        f, fieldnames=list(checks[0])
    )
    writer.writeheader()
    writer.writerows(checks)

(RESULTS / "gradient_check.txt").write_text(
    f"Initial loss: {initial_loss:.6f}\n"
    f"Expected loss: {np.log(10):.6f}\n"
    f"Gradient checks passed: {passed}/10\n"
    f"Required: at least 9/10\n"
)

if passed < 9:
    raise RuntimeError(
        "Gradient check failed. "
        "Training stopped."
    )

print("Gradient check PASSED.")


def train_network(width, learning_rate):

    np.random.seed(42)
    network = NeuralNetwork(width)

    losses = []

    batch_size = 32
    epochs = 50

    for epoch in range(epochs):

        indices = np.random.permutation(
            len(X_train)
        )

        for start in range(
            0, len(indices), batch_size
        ):

            batch_indices = indices[
                start:start + batch_size
            ]

            X_batch = X_train[batch_indices]
            Y_batch = Y_train[batch_indices]

            probabilities, cache = network.forward(
                X_batch
            )

            gradients = network.backward(
                cache, Y_batch
            )

            network.update(
                gradients, learning_rate
            )

        train_probabilities, _ = network.forward(
            X_train
        )

        epoch_loss = network.loss(
            train_probabilities, Y_train
        )

        losses.append(epoch_loss)

        print(
            f"Width={width}, LR={learning_rate}, "
            f"Epoch={epoch + 1}/50, "
            f"Loss={epoch_loss:.6f}"
        )

    predictions = network.predict(X_test)

    accuracy = np.mean(
        predictions == y_test
    )

    return network, losses, accuracy


widths = [16, 64]
learning_rates = [0.001, 0.01, 0.1]

results = []

plt.figure(figsize=(11, 7))

for width in widths:

    for learning_rate in learning_rates:

        print(
            f"\nTraining width={width}, "
            f"learning rate={learning_rate}"
        )

        network, losses, accuracy = train_network(
            width, learning_rate
        )

        results.append({
            "width": width,
            "learning_rate": learning_rate,
            "test_accuracy": round(float(accuracy), 4),
            "final_train_loss": round(
                float(losses[-1]), 6
            )
        })

        plt.plot(
            range(1, 51),
            losses,
            label=f"W={width}, LR={learning_rate}"
        )

        print(
            f"Final test accuracy: {accuracy:.4f}"
        )


plt.xlabel("Epoch")
plt.ylabel("Cross-Entropy Loss")
plt.title("Lab 2: Neural Network Training Loss")
plt.legend(fontsize=8)
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(
    RESULTS / "loss_curves.png",
    dpi=150
)

plt.close()

print("\nLoss curves saved.")



with (RESULTS / "accuracy.csv").open(
    "w", newline=""
) as f:

    writer = csv.DictWriter(
        f, fieldnames=list(results[0])
    )

    writer.writeheader()
    writer.writerows(results)

print("Accuracy results saved.")


def median_time(function, repetitions=100):

    function()  

    timings = []

    for _ in range(repetitions):

        start = time.perf_counter_ns()

        function()

        end = time.perf_counter_ns()

        timings.append(
            (end - start) / 1_000_000
        )

    return statistics.median(timings)


print("\nMeasuring forward and backward timing...")

timing_rows = []

for width in widths:

    np.random.seed(42)
    timing_network = NeuralNetwork(width)

    X_batch = X_train[:32]
    Y_batch = Y_train[:32]

    _, timing_cache = timing_network.forward(
        X_batch
    )

    forward_ms = median_time(
        lambda: timing_network.forward(X_batch)
    )

    backward_ms = median_time(
        lambda: timing_network.backward(
            timing_cache, Y_batch
        )
    )

    timing_rows.append({
        "width": width,
        "forward_median_ms": round(forward_ms, 6),
        "backward_median_ms": round(backward_ms, 6)
    })

    print(
        f"Width {width}: "
        f"Forward={forward_ms:.6f} ms, "
        f"Backward={backward_ms:.6f} ms"
    )


with (RESULTS / "timing.csv").open(
    "w", newline=""
) as f:

    writer = csv.DictWriter(
        f, fieldnames=list(timing_rows[0])
    )

    writer.writeheader()
    writer.writerows(timing_rows)


timing_text = "\n".join(
    f"Width {row['width']}\n"
    f"Forward median: {row['forward_median_ms']} ms\n"
    f"Backward median: {row['backward_median_ms']} ms\n"
    for row in timing_rows
)

(RESULTS / "timing.txt").write_text(
    timing_text
)



best = max(
    results,
    key=lambda row: row["test_accuracy"]
)

accuracy_table = "\n".join(
    f"| {r['width']} | {r['learning_rate']} | "
    f"{r['test_accuracy']:.4f} | "
    f"{r['final_train_loss']:.6f} |"
    for r in results
)

timing_table = "\n".join(
    f"| {r['width']} | "
    f"{r['forward_median_ms']:.6f} | "
    f"{r['backward_median_ms']:.6f} |"
    for r in timing_rows
)

timing_explanations = []

for row in timing_rows:

    if (
        row["backward_median_ms"]
        > row["forward_median_ms"]
    ):
        dominant = "backward propagation"
    else:
        dominant = "forward propagation"

    timing_explanations.append(
        f"For width {row['width']}, "
        f"{dominant} took longer in the measured runs."
    )

timing_explanation = " ".join(
    timing_explanations
)

report = f"""# Lab 2: Neural Network From Scratch

## Goal

The goal of this lab was to build and train a
two-hidden-layer neural network using NumPy.

The network was tested using the handwritten
digits dataset.

## Method

The dataset was loaded using load_digits.

Input values were divided by 16.

The dataset was divided into 70% training
and 30% testing data using stratification
and random_state=42.

The network architecture was:

64 inputs -> W -> ReLU -> W -> ReLU
-> 10 outputs -> Softmax

The hidden widths tested were 16 and 64.

Weights were initialized from a normal
distribution with mean 0 and standard
deviation 0.01.

Cross-entropy was used as the loss function.

Backpropagation was implemented manually
using the chain rule.

Mini-batch gradient descent was used with
a batch size of 32.

Each configuration was trained for 50 epochs.

Learning rates tested:

0.001, 0.01, 0.1

## Forward Propagation

The network uses matrix multiplication,
bias addition, ReLU activation and softmax.

The output contains probabilities for
10 digit classes.

The forward output shape and probability
sum checks passed.

Initial loss: {initial_loss:.6f}

Expected initial loss: {np.log(10):.6f}

## Gradient Checking

Numerical gradients were calculated using
the central difference method.

Epsilon: 1e-5

Relative error threshold: 1e-6

Gradient checks passed: {passed}/10

Required: at least 9/10.

## Training Results

| Width | Learning Rate | Test Accuracy | Final Loss |
|---|---|---|---|
{accuracy_table}

The loss curves are saved in:

results/loss_curves.png

## Timing Results

Each operation was warmed up once and
measured 100 times.

The median timing is reported in milliseconds.

| Width | Forward ms | Backward ms |
|---|---|---|
{timing_table}

{timing_explanation}

Forward propagation calculates the outputs
using matrix multiplication.

Backward propagation calculates gradients
through the chain rule and additional
matrix multiplications.

The measured difference depends on the
network width and the computer.

## Three Conclusions

1. The highest test accuracy was
{best['test_accuracy']:.4f}, achieved with
width {best['width']} and learning rate
{best['learning_rate']}.

2. The six configurations produced different
training losses and test accuracies.
The learning rate and hidden-layer width
affected the training results.

3. Forward and backward propagation both
require matrix multiplication. The timing
measurements show their computational costs
for the two hidden-layer widths.

## Files Generated

- results/versions.txt
- results/gradient_check.txt
- results/gradient_check.csv
- results/accuracy.csv
- results/loss_curves.png
- results/timing.txt
- results/timing.csv
"""

report_path = ROOT / "lab02" / "report.md"

report_path.write_text(
    report, encoding="utf-8"
)


print("\n================================")
print("LAB 2 COMPLETED SUCCESSFULLY")
print("================================")

print("\nGradient checks passed:", passed, "/10")

print("\nTraining Results:")

for row in results:
    print(row)

print("\nBest configuration:")
print(best)

print("\nResults saved to:")
print(RESULTS)

print("\nReport saved to:")
print(report_path)
