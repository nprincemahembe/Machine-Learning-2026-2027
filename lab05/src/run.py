
import sys
import time
import csv
import json
import platform
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

ROOT = Path(__file__).resolve().parents[2]
LAB = ROOT / "lab05"
R = LAB / "results"
R.mkdir(parents=True, exist_ok=True)

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(1)

print("\nLAB 5: FRAMEWORKS AND AUTOMATIC DIFFERENTIATION")
print("=" * 60)



class Value:
    def __init__(self, data, parents=(), op=""):
        self.data = float(data)
        self.grad = 0.0
        self.parents = tuple(parents)
        self.op = op
        self._backward = lambda: None

    def __repr__(self):
        return f"Value(data={self.data:.6f}, grad={self.grad:.6f})"

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = backward
        return out

    __radd__ = __add__

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = backward
        return out

    __rmul__ = __mul__

    def __pow__(self, power):
        out = Value(self.data ** power, (self,), f"**{power}")

        def backward():
            self.grad += (
                power * self.data ** (power - 1) * out.grad
            )

        out._backward = backward
        return out

    def __truediv__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return self * other ** -1

    def __rtruediv__(self, other):
        return Value(other) / self

    def relu(self):
        out = Value(max(0.0, self.data), (self,), "relu")

        def backward():
            self.grad += (self.data > 0) * out.grad

        out._backward = backward
        return out

    def exp(self):
        value = np.exp(self.data)
        out = Value(value, (self,), "exp")

        def backward():
            self.grad += value * out.grad

        out._backward = backward
        return out

    def log(self):
        out = Value(np.log(self.data), (self,), "log")

        def backward():
            self.grad += out.grad / self.data

        out._backward = backward
        return out

    def sigmoid(self):
       
        if self.data >= 0:
            result = 1 / (1 + np.exp(-self.data))
        else:
            e = np.exp(self.data)
            result = e / (1 + e)

        out = Value(result, (self,), "sigmoid")

        def backward():
            self.grad += result * (1 - result) * out.grad

        out._backward = backward
        return out

    def backward(self):
        order = []
        visited = set()

        def visit(node):
            if id(node) in visited:
                return

            visited.add(id(node))

            for parent in node.parents:
                visit(parent)

            order.append(node)

        visit(self)

      
        for node in order:
            node.grad = 0.0

        self.grad = 1.0

        for node in reversed(order):
            node._backward()


print("\n[1/7] Automatic differentiation engine created.")


print("\n[2/7] Validating gradients...")

def scalar_expression(a, b, c):
    return ((a * b + c) ** 2).relu()


def numerical_gradient(values, index, epsilon=1e-5):
    plus = values.copy()
    minus = values.copy()

    plus[index] += epsilon
    minus[index] -= epsilon

    f_plus = max(0.0, (plus[0] * plus[1] + plus[2]) ** 2)
    f_minus = max(0.0, (minus[0] * minus[1] + minus[2]) ** 2)

    return (f_plus - f_minus) / (2 * epsilon)


test_cases = [
    [1.2, -0.8, 2.1],
    [0.5, 1.3, 0.7],
    [-1.1, 0.4, 2.0],
    [2.0, -0.3, 1.5],
    [0.8, 0.9, -0.2]
]

gradient_rows = []

for case_number, values in enumerate(test_cases, 1):
    a, b, c = [Value(v) for v in values]

    result = scalar_expression(a, b, c)
    result.backward()

    torch_values = [
        torch.tensor(
            v, dtype=torch.float64, requires_grad=True
        )
        for v in values
    ]

    torch_result = scalar_expression(*torch_values)
    torch_result.backward()

    for i, name in enumerate(["a", "b", "c"]):
        manual_grad = [a, b, c][i].grad
        torch_grad = torch_values[i].grad.item()
        numerical_grad = numerical_gradient(values, i)

        gradient_rows.append({
            "test_case": case_number,
            "variable": name,
            "manual_gradient": manual_grad,
            "torch_gradient": torch_grad,
            "numerical_gradient": numerical_grad,
            "torch_difference": abs(manual_grad - torch_grad),
            "numerical_difference": abs(
                manual_grad - numerical_grad
            )
        })

with (R / "gradient_validation.csv").open(
    "w", newline="", encoding="utf-8"
) as file:
    writer = csv.DictWriter(
        file, fieldnames=gradient_rows[0].keys()
    )
    writer.writeheader()
    writer.writerows(gradient_rows)

max_torch_diff = max(
    row["torch_difference"] for row in gradient_rows
)

max_numerical_diff = max(
    row["numerical_difference"] for row in gradient_rows
)

assert max_torch_diff < 1e-6
assert max_numerical_diff < 1e-6

print("Maximum PyTorch difference:", max_torch_diff)
print("Maximum numerical difference:", max_numerical_diff)
print("Gradient validation: PASSED")



print("\n[3/7] Preparing make_moons dataset...")

X, y = make_moons(
    n_samples=1000,
    noise=0.1,
    random_state=SEED
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=SEED,
    stratify=y
)

x_train = torch.tensor(X_train, dtype=torch.float32)
t_train = torch.tensor(
    y_train.reshape(-1, 1), dtype=torch.float32
)

x_test = torch.tensor(X_test, dtype=torch.float32)

print("Training samples:", len(X_train))
print("Test samples:", len(X_test))



print("\n[4/7] Training manual autograd model...")

class ManualNetwork:
    def __init__(self):
        rng = np.random.default_rng(SEED)

        self.w1 = [
            [Value(v) for v in row]
            for row in rng.normal(0, 0.5, (2, 8))
        ]

        self.b1 = [Value(0) for _ in range(8)]

        self.w2 = [
            Value(v) for v in rng.normal(0, 0.5, 8)
        ]

        self.b2 = Value(0)

    def parameters(self):
        return (
            [p for row in self.w1 for p in row]
            + self.b1
            + self.w2
            + [self.b2]
        )

    def forward(self, row):
        hidden = []

        for j in range(8):
            z = self.b1[j]

            for i in range(2):
                z = z + row[i] * self.w1[i][j]

            hidden.append(z.relu())

        output = self.b2

        for j in range(8):
            output = output + hidden[j] * self.w2[j]

        return output

    def predict(self, row):
        return self.forward(row).sigmoid().data


manual_model = ManualNetwork()

def manual_training_step(row, label, learning_rate=0.05):
    logit = manual_model.forward(row)

    
    loss = (
        logit.relu()
        - logit * float(label)
        + ((-logit.relu()) + (-logit).relu()).exp().log()
    )

  
    abs_logit = logit.relu() + (-logit).relu()
    loss = (
        logit.relu()
        - logit * float(label)
        + (1 + (-abs_logit).exp()).log()
    )

    loss.backward()

    for parameter in manual_model.parameters():
        parameter.data -= learning_rate * parameter.grad

    return loss.data


manual_history = []


manual_X = X_train[:160]
manual_y = y_train[:160]

for epoch in range(25):
    losses = []

    for row, label in zip(manual_X, manual_y):
        losses.append(
            manual_training_step(row, label)
        )

    mean_loss = float(np.mean(losses))
    manual_history.append(mean_loss)

    if (epoch + 1) % 5 == 0:
        print(
            f"Manual epoch {epoch + 1}/25 "
            f"loss={mean_loss:.4f}"
        )

manual_predictions = np.array([
    int(manual_model.predict(row) >= 0.5)
    for row in X_test
])

manual_accuracy = accuracy_score(
    y_test, manual_predictions
)

print(f"Manual test accuracy: {manual_accuracy:.4f}")



print("\n[5/7] Training PyTorch model...")

torch_model = nn.Sequential(
    nn.Linear(2, 8),
    nn.ReLU(),
    nn.Linear(8, 1)
)

optimizer = torch.optim.SGD(
    torch_model.parameters(), lr=0.1
)

criterion = nn.BCEWithLogitsLoss()

torch_history = []

for epoch in range(150):
    optimizer.zero_grad()

    logits = torch_model(x_train)
    loss = criterion(logits, t_train)

    loss.backward()
    optimizer.step()

    torch_history.append(loss.item())

torch_model.eval()

with torch.no_grad():
    probabilities = torch.sigmoid(
        torch_model(x_test)
    )

    predictions = (
        probabilities.numpy().ravel() >= 0.5
    ).astype(int)

torch_accuracy = accuracy_score(
    y_test, predictions
)

print(f"PyTorch test accuracy: {torch_accuracy:.4f}")


print("\n[6/7] Benchmarking framework performance...")

def benchmark(fn, runs=5, warmup=10):
    for _ in range(warmup):
        fn()

    measurements = []

    for _ in range(runs):
        start = time.perf_counter()
        fn()
        measurements.append(
            (time.perf_counter() - start) * 1000
        )

    return float(np.median(measurements))



graph_input = x_train[:1].clone().requires_grad_(True)
graph_output = torch_model(graph_input)
graph_loss = graph_output.sum()

graph_lines = [
    "PyTorch autograd computation graph",
    f"Output grad_fn: {graph_loss.grad_fn}",
    ""
]

def inspect_graph(fn, depth=0, visited=None):
    if visited is None:
        visited = set()

    if fn is None or id(fn) in visited:
        return

    visited.add(id(fn))
    graph_lines.append(
        "  " * depth + type(fn).__name__
    )

    for next_fn, _ in fn.next_functions:
        inspect_graph(next_fn, depth + 1, visited)

inspect_graph(graph_loss.grad_fn)

(R / "pytorch_graph.txt").write_text(
    "\n".join(graph_lines),
    encoding="utf-8"
)


example = torch.randn(16, 2)
traced_model = torch.jit.trace(torch_model, example)
traced_model.eval()

traced_model.save(
    str(R / "traced_model.pt")
)


bench_input = x_train[:16]

with torch.inference_mode():
    eager_ms = benchmark(
        lambda: torch_model(bench_input)
    )

    traced_ms = benchmark(
        lambda: traced_model(bench_input)
    )

single_row = X_train[0]
single_label = y_train[0]

manual_step_ms = benchmark(
    lambda: manual_training_step(
        single_row, single_label
    ),
    warmup=3
)

single_x = x_train[:1]
single_y = t_train[:1]

torch_model.train()

def torch_training_step():
    optimizer.zero_grad()
    logits = torch_model(single_x)
    loss = criterion(logits, single_y)
    loss.backward()
    optimizer.step()

torch_step_ms = benchmark(
    torch_training_step,
    warmup=3
)

torch_model.eval()


matrix = torch.randn(512, 512)
noncontiguous = matrix.t()
contiguous = noncontiguous.contiguous()


assert torch.equal(noncontiguous, contiguous)

with torch.inference_mode():
    noncontiguous_ms = benchmark(
        lambda: noncontiguous @ matrix
    )

    contiguous_ms = benchmark(
        lambda: contiguous @ matrix
    )

timings = [
    {"metric": "manual_training_step", "median_ms": manual_step_ms},
    {"metric": "torch_training_step", "median_ms": torch_step_ms},
    {"metric": "eager_forward", "median_ms": eager_ms},
    {"metric": "traced_forward", "median_ms": traced_ms},
    {"metric": "noncontiguous_matmul", "median_ms": noncontiguous_ms},
    {"metric": "contiguous_matmul", "median_ms": contiguous_ms}
]

with (R / "timings.csv").open(
    "w", newline="", encoding="utf-8"
) as file:
    writer = csv.DictWriter(
        file, fieldnames=["metric", "median_ms"]
    )
    writer.writeheader()
    writer.writerows(timings)

tensor_info = {
    "shape": list(noncontiguous.shape),
    "dtype": str(noncontiguous.dtype),
    "noncontiguous_strides": list(noncontiguous.stride()),
    "contiguous_strides": list(contiguous.stride()),
    "noncontiguous_is_contiguous": noncontiguous.is_contiguous(),
    "contiguous_is_contiguous": contiguous.is_contiguous()
}

(R / "tensor_properties.json").write_text(
    json.dumps(tensor_info, indent=2),
    encoding="utf-8"
)

print(f"Eager forward: {eager_ms:.6f} ms")
print(f"Traced forward: {traced_ms:.6f} ms")
print(f"Manual training step: {manual_step_ms:.6f} ms")
print(f"PyTorch training step: {torch_step_ms:.6f} ms")
print(f"Non-contiguous matmul: {noncontiguous_ms:.6f} ms")
print(f"Contiguous matmul: {contiguous_ms:.6f} ms")


print("\n[7/7] Generating plots and report...")

plt.figure(figsize=(8, 5))
plt.plot(
    range(1, len(manual_history) + 1),
    manual_history,
    label="Manual autograd"
)
plt.xlabel("Epoch")
plt.ylabel("Mean BCE loss")
plt.title("Manual Autograd Training")
plt.legend()
plt.tight_layout()
plt.savefig(
    R / "manual_training_curve.png",
    dpi=150
)
plt.close()

plt.figure(figsize=(8, 5))
plt.plot(
    range(1, len(torch_history) + 1),
    torch_history,
    label="PyTorch"
)
plt.xlabel("Epoch")
plt.ylabel("BCE loss")
plt.title("PyTorch Training")
plt.legend()
plt.tight_layout()
plt.savefig(
    R / "torch_training_curve.png",
    dpi=150
)
plt.close()

plt.figure(figsize=(8, 5))
plt.bar(
    ["Eager", "Traced"],
    [eager_ms, traced_ms]
)
plt.ylabel("Median time (ms)")
plt.title("Eager vs Traced Execution")
plt.tight_layout()
plt.savefig(
    R / "execution_comparison.png",
    dpi=150
)
plt.close()

plt.figure(figsize=(8, 5))
plt.bar(
    ["Non-contiguous", "Contiguous"],
    [noncontiguous_ms, contiguous_ms]
)
plt.ylabel("Median time (ms)")
plt.title("Tensor Contiguity Performance")
plt.tight_layout()
plt.savefig(
    R / "contiguity_comparison.png",
    dpi=150
)
plt.close()

versions = {
    "python": sys.version.split()[0],
    "platform": platform.platform(),
    "numpy": np.__version__,
    "torch": torch.__version__,
    "device": "CPU",
    "seed": SEED
}

(R / "versions.json").write_text(
    json.dumps(versions, indent=2),
    encoding="utf-8"
)

if traced_ms < eager_ms:
    execution_interpretation = (
        "Tracing was faster in this experiment. "
        "It may reduce Python execution overhead "
        "by representing operations in a reusable graph."
    )
else:
    execution_interpretation = (
        "Tracing was not faster in this experiment. "
        "For a very small network, the saved overhead "
        "may be too small to outweigh execution costs."
    )

if noncontiguous_ms > contiguous_ms:
    contiguity_interpretation = (
        "The non-contiguous view was slower in this test. "
        "Its memory layout may require less efficient access."
    )
else:
    contiguity_interpretation = (
        "The non-contiguous view was not slower in this test. "
        "Performance depends on the operation, memory layout "
        "and underlying matrix multiplication implementation."
    )

report = f"""# Lab 5: Frameworks and Automatic Differentiation

## 1. Goal

The aim of this lab was to understand computation graphs,
automatic differentiation, execution modes and tensor memory
representation.

## 2. Scalar Automatic Differentiation

I implemented a Value class that stores a scalar value,
its gradient, parent nodes and the local backward rule.

Addition, multiplication, powers and ReLU build a computation
graph during the forward pass.

The backward method uses a topological ordering of the graph.
It starts with output gradient 1 and visits nodes in reverse
order, accumulating gradients instead of overwriting them.

## 3. Gradient Validation

Expression tested:

((a * b + c) ** 2).relu()

Maximum difference compared with PyTorch:
{max_torch_diff:.12g}

Maximum difference compared with numerical gradients:
{max_numerical_diff:.12g}

Both differences were below 1e-6.

Result: PASSED.

Detailed evidence: results/gradient_validation.csv

## 4. Training on make_moons

Dataset: 1,000 samples, noise 0.1.

Architecture: 2 -> 8 -> 1, with ReLU hidden activation,
sigmoid output and binary cross-entropy loss.

Manual engine:
- Training subset: 160 observations
- Epochs: 25
- Test accuracy: {manual_accuracy:.4f}

PyTorch:
- Training observations: {len(X_train)}
- Epochs: 150
- Test accuracy: {torch_accuracy:.4f}

The two models use the same architecture but different
training schedules. Their accuracies are not a controlled
comparison of framework quality.

## 5. Training Step Speed

Manual autograd median:
{manual_step_ms:.6f} ms

PyTorch median:
{torch_step_ms:.6f} ms

Both timings measure a single-observation forward,
backward and parameter-update step.

Each median is calculated from five timed runs
after warm-up.

The manual implementation constructs many Python objects,
whereas PyTorch uses optimized tensor operations and
automatic differentiation.

## 6. PyTorch Computation Graph

The computation graph was inspected using grad_fn
and next_functions.

The inspected nodes are saved in:
results/pytorch_graph.txt

PyTorch automatically tracks differentiable operations
and uses the graph to calculate gradients.

## 7. Eager vs Traced Execution

Eager median:
{eager_ms:.6f} ms

Traced median:
{traced_ms:.6f} ms

{execution_interpretation}

Compilation or tracing can help when repeated operations
allow the framework to reduce Python overhead or optimize
execution.

However, tracing does not guarantee a speed improvement,
especially for small models. Compilation also has an
upfront cost, so it is more useful when a model runs
repeatedly enough to recover that cost.

These timings use torch.jit.trace on CPU rather than
torch.compile. They are specific to this experiment.

## 8. Tensor Representation

Tensor shape: {tensor_info['shape']}
Tensor dtype: {tensor_info['dtype']}

Non-contiguous strides:
{tensor_info['noncontiguous_strides']}

Contiguous strides:
{tensor_info['contiguous_strides']}

Non-contiguous matrix multiplication:
{noncontiguous_ms:.6f} ms

Contiguous matrix multiplication:
{contiguous_ms:.6f} ms

{contiguity_interpretation}

A transposed tensor can share the same underlying data
but have different strides. This affects how elements
are accessed in memory.

## 9. Three Conclusions

1. Automatic differentiation works by recording operations
in a computation graph and applying the chain rule backward.

2. PyTorch simplifies gradient calculation and provides
optimized operations. The scalar engine is useful for
learning but is not designed for large-scale training.

3. Execution mode and tensor layout can affect performance.
Tracing is not always faster, and a non-contiguous tensor
is not always slower. Measurements are needed.

## 10. Reproducibility

Random seed: {SEED}
PyTorch version: {torch.__version__}
Python version: {sys.version.split()[0]}
Device: CPU

Run from the repository root:

python lab05/src/run.py

## 11. Evidence

- results/gradient_validation.csv
- results/pytorch_graph.txt
- results/timings.csv
- results/tensor_properties.json
- results/manual_training_curve.png
- results/torch_training_curve.png
- results/execution_comparison.png
- results/contiguity_comparison.png
- results/traced_model.pt
- results/versions.json

## 12. Reference

Vijay Janapa Reddi, Introduction to Machine Learning
Systems: Principles and Practices of Engineering
Artificially Intelligent Systems, Chapter 7,
Sections 7.3, 7.4, 7.6, 7.7 and 7.8.
"""

(LAB / "report.md").write_text(
    report,
    encoding="utf-8"
)

print("\n" + "=" * 60)
print("LAB 5 COMPLETED SUCCESSFULLY")
print("=" * 60)

print("\nGradient check: PASSED")
print(f"Manual accuracy: {manual_accuracy:.4f}")
print(f"PyTorch accuracy: {torch_accuracy:.4f}")
print(f"Eager: {eager_ms:.6f} ms")
print(f"Traced: {traced_ms:.6f} ms")

print("\nResults:", R)
print("Report:", LAB / "report.md")
