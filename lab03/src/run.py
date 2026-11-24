
import sys
import csv
import time
import random
import statistics
import warnings
from pathlib import Path
from importlib.metadata import version, PackageNotFoundError

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torchinfo import summary
from thop import profile


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "lab03" / "results"
DATA = ROOT / "data"

RESULTS.mkdir(parents=True, exist_ok=True)
DATA.mkdir(parents=True, exist_ok=True)

SEED = 42
BATCH_SIZE = 128
EPOCHS = 5
LEARNING_RATE = 0.001

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Starting Machine Learning Lab 3...")
print("Device:", DEVICE)


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

    lines = [
        f"Python: {sys.version.split()[0]}",
        f"Device: {DEVICE}"
    ]

    for package in packages:
        try:
            installed = version(package)
        except PackageNotFoundError:
            installed = "NOT INSTALLED"

        lines.append(f"{package}: {installed}")

    (RESULTS / "versions.txt").write_text(
        "\n".join(lines), encoding="utf-8"
    )


save_versions()


def save_csv(filename, rows):
    if not rows:
        return

    with (RESULTS / filename).open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file, fieldnames=list(rows[0].keys())
        )
        writer.writeheader()
        writer.writerows(rows)


print("\nLoading Fashion-MNIST dataset...")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        (0.2860,),
        (0.3530,)
    )
])

full_train = datasets.FashionMNIST(
    root=str(DATA),
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.FashionMNIST(
    root=str(DATA),
    train=False,
    download=True,
    transform=transform
)

train_dataset, val_dataset = random_split(
    full_train,
    [55000, 5000],
    generator=torch.Generator().manual_seed(SEED)
)

pin_memory = DEVICE.type == "cuda"

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=pin_memory
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=pin_memory
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=pin_memory
)

print("Training samples:", len(train_dataset))
print("Validation samples:", len(val_dataset))
print("Testing samples:", len(test_dataset))


class MLP(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.network(x)


class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(
                1, 16, kernel_size=3, padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(
                16, 32, kernel_size=3, padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 10)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


class AttentionModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.patch_size = 7
        self.embedding_dim = 64

       
        self.patch_projection = nn.Linear(49, 64)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=64,
            nhead=4,
            dim_feedforward=128,
            dropout=0.1,
            batch_first=True
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=2,
            enable_nested_tensor=False
        )

        self.classifier = nn.Linear(64, 10)

    def forward(self, x):
        batch_size = x.shape[0]

       
        patches = x.unfold(2, 7, 7).unfold(3, 7, 7)

        patches = patches.contiguous().reshape(
            batch_size, 16, 49
        )

        x = self.patch_projection(patches)

        x = self.encoder(x)

      
        x = x.mean(dim=1)

        return self.classifier(x)



models = {
    "MLP": MLP(),
    "CNN": CNN(),
    "Attention": AttentionModel()
}

print("\nThree neural network models created.")


print("\nCounting model parameters...")

parameter_rows = []
summary_lines = []

for name, model in models.items():
    model.eval()

    info = summary(
        model,
        input_size=(1, 1, 28, 28),
        device="cpu",
        verbose=0
    )

    measured_parameters = int(
        info.total_params
    )

    manual_count = sum(
        p.numel() for p in model.parameters()
    )

    parameter_rows.append({
        "model": name,
        "torchinfo_parameters": measured_parameters,
        "parameter_sum_check": manual_count,
        "within_target_100k_150k":
            100000 <= measured_parameters <= 150000
    })

    summary_lines.append(
        f"\n{'=' * 55}\n"
        f"{name}\n"
        f"{'=' * 55}\n"
        f"{info}\n"
    )

    print(
        f"{name}: {measured_parameters:,} parameters"
    )

(RESULTS / "model_summary.txt").write_text(
    "\n".join(summary_lines),
    encoding="utf-8"
)

save_csv(
    "parameter_counts.csv",
    parameter_rows
)



def linear_flops(in_features, out_features, tokens=1):
    return 2 * tokens * in_features * out_features


def conv_flops(
    kernel, in_channels, out_channels,
    height, width
):
    return (
        2 * kernel * kernel
        * in_channels * out_channels
        * height * width
    )


def attention_layer_flops(seq=16, dim=64, ff=128):

    qkv = 2 * seq * dim * (3 * dim)

   
    scores = 2 * seq * seq * dim

   
    weighted_values = 2 * seq * seq * dim

    output_projection = 2 * seq * dim * dim

    feed_forward = (
        2 * seq * dim * ff
        + 2 * seq * ff * dim
    )

    return (
        qkv + scores + weighted_values
        + output_projection + feed_forward
    )



mlp_flops = (
    linear_flops(784, 256)
    + linear_flops(256, 128)
    + linear_flops(128, 10)
)


cnn_flops = (
    conv_flops(3, 1, 16, 28, 28)
    + conv_flops(3, 16, 32, 14, 14)
    + linear_flops(32 * 7 * 7, 10)
)


attention_flops = (
    linear_flops(49, 64, tokens=16)
    + 2 * attention_layer_flops()
    + linear_flops(64, 10)
)

hand_flops = {
    "MLP": mlp_flops,
    "CNN": cnn_flops,
    "Attention": attention_flops
}

print("\nHand-calculated FLOPs:")

for name, value in hand_flops.items():
    print(f"{name}: {value:,.0f} FLOPs")


print("\nRunning THOP profiling...")

flops_rows = []

for name, model in models.items():
    model.eval()

    dummy = torch.randn(1, 1, 28, 28)

    thop_macs = None
    measured_flops = None
    profiler_note = ""

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            macs, _ = profile(
                model,
                inputs=(dummy,),
                verbose=False
            )

        thop_macs = int(macs)

        measured_flops = 2 * thop_macs

        if name == "Attention":
            profiler_note = (
                "THOP may omit Transformer attention "
                "and functional operations."
            )
        else:
            profiler_note = (
                "THOP coverage depends on supported layers."
            )

    except Exception as error:
        profiler_note = (
            "THOP could not fully profile this model: "
            + str(error)[:160]
        )

    hand_value = hand_flops[name]

    if measured_flops is not None:
        difference = measured_flops - hand_value
        difference_percent = (
            100 * difference / hand_value
        )
    else:
        difference = ""
        difference_percent = ""

    flops_rows.append({
        "model": name,
        "hand_calculated_flops": hand_value,
        "thop_macs": (
            thop_macs if thop_macs is not None else ""
        ),
        "thop_estimated_flops": (
            measured_flops
            if measured_flops is not None else ""
        ),
        "difference_flops": difference,
        "difference_percent": difference_percent,
        "profiler_note": profiler_note
    })

    print(
        f"{name}: "
        f"Hand={hand_value:,}, "
        f"THOP estimate={measured_flops}"
    )

save_csv(
    "flops_comparison.csv",
    flops_rows
)


criterion = nn.CrossEntropyLoss()


def evaluate(model, loader):
    model.eval()

    correct = 0
    total = 0
    total_loss = 0.0

    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * len(labels)

            predictions = outputs.argmax(dim=1)

            correct += (
                predictions == labels
            ).sum().item()

            total += len(labels)

    return (
        total_loss / total,
        correct / total
    )


def train_model(name, model):
    model = model.to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    history = []

    print(f"\nTraining {name}...")

    for epoch in range(1, EPOCHS + 1):
        model.train()

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        start = time.perf_counter()

        for images, labels in train_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad(set_to_none=True)

            outputs = model(images)

            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(labels)

            total_correct += (
                outputs.argmax(dim=1) == labels
            ).sum().item()

            total_samples += len(labels)

        train_loss = total_loss / total_samples
        train_accuracy = total_correct / total_samples

        val_loss, val_accuracy = evaluate(
            model, val_loader
        )

        duration = time.perf_counter() - start

        history.append({
            "model": name,
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "train_accuracy": round(
                train_accuracy, 4
            ),
            "validation_loss": round(val_loss, 6),
            "validation_accuracy": round(
                val_accuracy, 4
            ),
            "epoch_seconds": round(duration, 3)
        })

        print(
            f"{name} | Epoch {epoch}/{EPOCHS} | "
            f"Train Acc={train_accuracy:.4f} | "
            f"Val Acc={val_accuracy:.4f} | "
            f"Loss={train_loss:.4f} | "
            f"Time={duration:.1f}s"
        )

    test_loss, test_accuracy = evaluate(
        model, test_loader
    )

    print(
        f"{name} test accuracy: "
        f"{test_accuracy:.4f}"
    )

    return history, test_loss, test_accuracy


accuracy_rows = []
training_history = []

for name, model in models.items():
    history, test_loss, test_accuracy = train_model(
        name, model
    )

    training_history.extend(history)

    parameters = next(
        row["torchinfo_parameters"]
        for row in parameter_rows
        if row["model"] == name
    )

    accuracy_rows.append({
        "model": name,
        "test_accuracy": round(test_accuracy, 4),
        "test_loss": round(test_loss, 6),
        "parameters": parameters
    })

save_csv(
    "accuracy.csv",
    accuracy_rows
)

save_csv(
    "training_history.csv",
    training_history
)


def synchronize_device():
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()


def measure_latency(model, batch_size, runs):
    model.eval()

    sample = torch.randn(
        batch_size, 1, 28, 28,
        device=DEVICE
    )

    times = []

    with torch.inference_mode():

        
        for _ in range(3):
            model(sample)

        synchronize_device()

        for _ in range(runs):
            synchronize_device()

            start = time.perf_counter_ns()

            model(sample)

            synchronize_device()

            end = time.perf_counter_ns()

            times.append(
                (end - start) / 1_000_000
            )

    return statistics.median(times)


print("\nMeasuring inference latency...")

latency_rows = []

for name, model in models.items():

    for batch_size, runs in [
        (1, 100),
        (128, 20)
    ]:

        latency = measure_latency(
            model, batch_size, runs
        )

        latency_rows.append({
            "model": name,
            "batch_size": batch_size,
            "runs": runs,
            "median_latency_ms": round(
                latency, 6
            )
        })

        print(
            f"{name} | Batch={batch_size} | "
            f"Median={latency:.6f} ms"
        )

save_csv(
    "latency.csv",
    latency_rows
)


print("\nGenerating trade-off plots...")

plt.figure(figsize=(9, 6))

for row in accuracy_rows:
    plt.scatter(
        row["parameters"],
        row["test_accuracy"] * 100,
        s=110
    )

    plt.annotate(
        row["model"],
        (
            row["parameters"],
            row["test_accuracy"] * 100
        ),
        xytext=(6, 6),
        textcoords="offset points"
    )

plt.xlabel("Number of Parameters")
plt.ylabel("Test Accuracy (%)")
plt.title("Lab 3: Accuracy vs Parameters")
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(
    RESULTS / "accuracy_vs_parameters.png",
    dpi=150
)
plt.close()


plt.figure(figsize=(9, 6))

for row in accuracy_rows:
    name = row["model"]

    latency = next(
        item["median_latency_ms"]
        for item in latency_rows
        if item["model"] == name
        and item["batch_size"] == 1
    )

    plt.scatter(
        latency,
        row["test_accuracy"] * 100,
        s=110
    )

    plt.annotate(
        name,
        (latency, row["test_accuracy"] * 100),
        xytext=(6, 6),
        textcoords="offset points"
    )

plt.xlabel("Batch-1 Median Latency (ms)")
plt.ylabel("Test Accuracy (%)")
plt.title("Lab 3: Accuracy vs Inference Latency")
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(
    RESULTS / "accuracy_vs_latency.png",
    dpi=150
)
plt.close()


plt.figure(figsize=(10, 6))

for name in models:
    rows = [
        row for row in training_history
        if row["model"] == name
    ]

    plt.plot(
        [row["epoch"] for row in rows],
        [row["validation_accuracy"] * 100 for row in rows],
        marker="o",
        label=name
    )

plt.xlabel("Epoch")
plt.ylabel("Validation Accuracy (%)")
plt.title("Lab 3: Validation Accuracy by Epoch")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(
    RESULTS / "training_curves.png",
    dpi=150
)
plt.close()


def markdown_table(rows, columns):
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(
        ["---"] * len(columns)
    ) + " |"

    body = [
        "| " + " | ".join(
            str(row[column]) for column in columns
        ) + " |"
        for row in rows
    ]

    return "\n".join(
        [header, separator] + body
    )


best_model = max(
    accuracy_rows,
    key=lambda row: row["test_accuracy"]
)

smallest_model = min(
    accuracy_rows,
    key=lambda row: row["parameters"]
)

batch1 = {
    row["model"]: row["median_latency_ms"]
    for row in latency_rows
    if row["batch_size"] == 1
}

fastest_model = min(
    batch1,
    key=batch1.get
)

parameter_table = markdown_table(
    parameter_rows,
    [
        "model",
        "torchinfo_parameters",
        "within_target_100k_150k"
    ]
)

flops_table = markdown_table(
    flops_rows,
    [
        "model",
        "hand_calculated_flops",
        "thop_macs",
        "thop_estimated_flops",
        "difference_percent"
    ]
)

accuracy_table = markdown_table(
    accuracy_rows,
    [
        "model",
        "test_accuracy",
        "test_loss",
        "parameters"
    ]
)

latency_table = markdown_table(
    latency_rows,
    [
        "model",
        "batch_size",
        "runs",
        "median_latency_ms"
    ]
)

report = f"""# Lab 3: Architectures and Their Costs

## 1. Goal

The goal of this lab was to implement and compare
three deep neural network architectures using PyTorch.

The models were an MLP, a CNN and an attention-based
Transformer.

The comparison focused on accuracy, parameter count,
computational cost, inference latency and memory access.

## 2. Dataset and Method

The Fashion-MNIST dataset was used.

The images were transformed using ToTensor and
Normalize((0.2860,), (0.3530,)).

The official training dataset was split into:

- 55,000 training images
- 5,000 validation images

The official test dataset contained 10,000 images.

The batch size was 128.

All models were trained for 5 epochs using Adam
with a learning rate of 0.001.

Random seed: 42.

Training device: {DEVICE}.

## 3. Model Architectures

### MLP

784 input features -> 256 neurons -> ReLU
-> 128 neurons -> ReLU -> 10 outputs.

### CNN

Conv2d(1,16,3,padding=1) -> ReLU -> MaxPool2d(2)
-> Conv2d(16,32,3,padding=1) -> ReLU
-> MaxPool2d(2) -> Flatten -> Linear(1568,10).

### Attention Model

The 28x28 image was divided into 16 patches
of size 7x7.

Each patch was projected into 64 features.

The model used two Transformer encoder layers
with d_model=64, nhead=4 and
dim_feedforward=128.

Mean pooling and a Linear(64,10) classifier
were used.

## 4. Parameter Counts

{parameter_table}

The manual gives a target of 100,000 to 150,000
parameters, but its specified architectures do
not all fall within that range.

The exact prescribed architectures were used
to preserve the laboratory design.

## 5. Hand-Calculated FLOPs

The representative formulas used were:

Linear layer:

FLOPs = 2 x input_features x output_features.

Convolution layer:

FLOPs = 2 x K x K x C_in x C_out x H_out x W_out.

Attention QKV projections:

FLOPs = 2 x sequence_length x d_model x 3 x d_model.

Attention score calculation:

FLOPs = 2 x sequence_length^2 x d_model.

The hand estimates also include the attention
weighted-value multiplication, output projection
and two feed-forward linear layers.

### Example calculations

MLP first linear layer:

2 x 784 x 256 = {linear_flops(784, 256):,} FLOPs.

CNN first convolution:

2 x 3 x 3 x 1 x 16 x 28 x 28
= {conv_flops(3, 1, 16, 28, 28):,} FLOPs.

Attention QKV projections for one encoder layer:

2 x 16 x 64 x 3 x 64
= {2 * 16 * 64 * 3 * 64:,} FLOPs.

## 6. Hand Calculations vs THOP

{flops_table}

THOP primarily reports multiply-accumulate operations
(MACs). The report approximates 1 MAC as 2 FLOPs.

The hand calculations focus on representative
matrix multiplication and convolution arithmetic.

They do not fully count bias addition, activation,
normalization, pooling, softmax or memory operations.

THOP also depends on which PyTorch operations it
supports. In particular, some Transformer attention
operations may be missing from its result.

Therefore, a difference between the hand estimates
and the profiler does not automatically mean that
the hand calculation is incorrect.

The Attention result should not be treated as a
complete measured FLOP count unless all operations
are supported by the profiler.

## 7. Test Accuracy

{accuracy_table}

## 8. Inference Latency

The models were tested with batch sizes 1 and 128.

Batch size 1 used 100 timed runs.

Batch size 128 used 20 timed runs.

Warm-up runs were completed before measurement.

The median time was recorded.

{latency_table}

The timings describe performance on the device
used for this experiment, not every possible device.

## 9. Architecture Trade-Offs

### MLP

The MLP uses fully connected layers.

These layers require dense matrix multiplications.

Flattening the image removes the explicit
two-dimensional spatial structure.

Its memory access involves reading dense
weight matrices and intermediate activations.

### CNN

The CNN uses small convolution filters that
are reused across different image positions.

This weight sharing can reduce the number
of parameters compared with dense layers.

Convolutions can reuse nearby image data,
although feature maps still require memory
access and intermediate storage.

The CNN is designed to recognize local
spatial patterns such as edges and shapes.

### Attention

The attention model divides images into patches
and allows the patches to exchange information.

It computes query, key and value projections
and attention matrices.

Attention score computation grows with the
square of the sequence length.

This introduces additional intermediate data
and memory traffic.

Attention can capture relationships between
distant patches, but its overhead may be
significant for small images.

### Cost-Effectiveness

The highest measured test accuracy was achieved
by {best_model['model']} with
{best_model['test_accuracy']:.4f} accuracy.

The smallest model by parameter count was
{smallest_model['model']}.

The fastest model for batch-1 inference on
this device was {fastest_model}.

These are different measurements, so the
most cost-effective architecture depends on
whether accuracy, model size or latency is
the main priority.

The graphs provide a visual comparison.

## 10. Trade-Off Plots

Accuracy against parameters:

![Accuracy vs Parameters](results/accuracy_vs_parameters.png)

Accuracy against latency:

![Accuracy vs Latency](results/accuracy_vs_latency.png)

Validation accuracy curves:

![Training Curves](results/training_curves.png)

## 11. Three Conclusions

1. In this experiment, {best_model['model']}
achieved the highest test accuracy of
{best_model['test_accuracy']:.4f}.

2. The model with the fewest parameters was
{smallest_model['model']}, while the fastest
batch-1 inference was measured for
{fastest_model}. This shows that parameter
count and latency are not the same thing.

3. The three architectures use different
computation and memory-access patterns.
The MLP uses dense connections, the CNN
reuses convolution filters, and attention
calculates relationships between patches.

## 12. Generated Evidence

- results/versions.txt
- results/model_summary.txt
- results/parameter_counts.csv
- results/flops_comparison.csv
- results/accuracy.csv
- results/training_history.csv
- results/latency.csv
- results/accuracy_vs_parameters.png
- results/accuracy_vs_latency.png
- results/training_curves.png


"""

report_path = ROOT / "lab03" / "report.md"

report_path.write_text(
    report,
    encoding="utf-8"
)



print("\n===================================")
print("LAB 3 COMPLETED SUCCESSFULLY")
print("===================================")

print("\nAccuracy results:")

for row in accuracy_rows:
    print(row)

print("\nParameter counts:")

for row in parameter_rows:
    print(row)

print("\nLatency results:")

for row in latency_rows:
    print(row)

print("\nBest accuracy model:")
print(best_model)

print("\nResults saved to:")
print(RESULTS)

print("\nReport saved to:")
print(report_path)
