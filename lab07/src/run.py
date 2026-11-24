import argparse
import copy
import io
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from torch import nn
from torch.nn.utils import prune
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from thop import profile as thop_profile

LAB = ROOT / "lab07"
R = LAB / "results"
R.mkdir(parents=True, exist_ok=True)

SEED = 42
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

METRICS = R / "compression_metrics.csv"


# ==================================================
# 1. REPRODUCIBILITY AND DATASET
# ==================================================

def seed_all(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_loaders():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.286,), (0.353,))
    ])

    train = datasets.FashionMNIST(
        str(ROOT / "data"),
        train=True,
        download=True,
        transform=transform
    )

    test = datasets.FashionMNIST(
        str(ROOT / "data"),
        train=False,
        download=True,
        transform=transform
    )

    generator = torch.Generator().manual_seed(SEED)

    dl = DataLoader(
        train,
        batch_size=128,
        shuffle=True,
        num_workers=1,
        persistent_workers=True,
        generator=generator
    )

    td = DataLoader(
        test,
        batch_size=256,
        shuffle=False,
        num_workers=1,
        persistent_workers=True
    )

    return dl, td


# ==================================================
# 2. MODEL ARCHITECTURES
# ==================================================

class Big(nn.Module):

    def __init__(self, c1=32, c2=64):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, c1, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(c1, c2, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(c2 * 7 * 7, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        return self.head(self.features(x))


class Small(nn.Module):

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        return self.head(self.features(x))


# ==================================================
# 3. TRAINING AND KNOWLEDGE DISTILLATION
# ==================================================

def fit(model, dl, epochs, tag, teacher=None):

    seed_all()
    dl.generator.manual_seed(SEED)

    model.to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.001
    )

    if teacher is not None:
        teacher.to(DEVICE)
        teacher.eval()

    history = []
    tracker = None
    energy_kwh = None

    try:
        from codecarbon import OfflineEmissionsTracker

        tracker = OfflineEmissionsTracker(
            country_iso_code="POL",
            project_name=tag,
            output_dir=str(R),
            output_file=f"{tag}_energy.csv",
            measure_power_secs=5,
            log_level="error"
        )

        tracker.start()

    except Exception as exc:
        print("Energy tracker unavailable:", exc)
        tracker = None

    start = time.perf_counter()

    try:
        for ep in range(epochs):

            model.train()
            total_loss = 0.0
            total_samples = 0

            for x, y in dl:

                x = x.to(DEVICE)
                y = y.to(DEVICE)

                optimizer.zero_grad()

                logits = model(x)

                ce_loss = F.cross_entropy(logits, y)
                loss = ce_loss

                if teacher is not None:

                    with torch.no_grad():
                        teacher_logits = teacher(x)

                    T = 4.0
                    alpha = 0.5

                    kd_loss = F.kl_div(
                        F.log_softmax(logits / T, dim=1),
                        F.softmax(
                            teacher_logits / T, dim=1
                        ),
                        reduction="batchmean"
                    )

                    loss = (
                        alpha * ce_loss
                        + (1 - alpha) * T * T * kd_loss
                    )

                loss.backward()
                optimizer.step()

                total_loss += loss.item() * len(y)
                total_samples += len(y)

            epoch_loss = total_loss / total_samples

            history.append({
                "epoch": ep + 1,
                "train_loss": epoch_loss
            })

            pd.DataFrame(history).to_csv(
                R / f"{tag}_history.csv",
                index=False
            )

            torch.save({
                "epoch": ep + 1,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict()
            }, R / f"{tag}_checkpoint.pt")

            print(
                f"{tag}: epoch {ep + 1}/{epochs} | "
                f"loss {epoch_loss:.4f}",
                flush=True
            )

    finally:

        if tracker is not None:

            try:
                tracker.stop()

                energy_file = R / f"{tag}_energy.csv"

                if energy_file.exists():
                    energy_data = pd.read_csv(energy_file)

                    if "energy_consumed" in energy_data:
                        energy_kwh = float(
                            energy_data[
                                "energy_consumed"
                            ].iloc[-1]
                        )

            except Exception as exc:
                print("Energy measurement error:", exc)

    seconds = time.perf_counter() - start

    return model, seconds, energy_kwh


# ==================================================
# 4. EVALUATION AND BENCHMARKING
# ==================================================

@torch.inference_mode()
def evaluate(model, td):

    model.eval()
    correct = 0
    total = 0

    for x, y in td:

        x = x.to(DEVICE)

        predictions = model(x).argmax(dim=1).cpu()

        correct += (predictions == y).sum().item()
        total += len(y)

    return correct / total


def serialized_size(state):

    buffer = io.BytesIO()
    torch.save(state, buffer)

    return len(buffer.getvalue())


@torch.inference_mode()
def measure(model):

    model.eval()

    x = torch.randn(
        1, 1, 28, 28, device=DEVICE
    )

    def sync():
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()

    # Warm-up
    for _ in range(30):
        model(x)

    sync()

    timings = []

    for _ in range(100):

        sync()
        start = time.perf_counter()

        model(x)

        sync()

        timings.append(
            (time.perf_counter() - start) * 1000
        )

    latency = float(np.median(timings))

    dense_state = {
        k: v.detach().cpu()
        for k, v in model.state_dict().items()
    }

    sparse_state = {}

    for key, tensor in dense_state.items():

        if tensor.ndim >= 2:
            sparse_state[key] = (
                tensor.to_sparse().coalesce()
            )
        else:
            sparse_state[key] = tensor

    dense_bytes = serialized_size(dense_state)
    sparse_bytes = serialized_size(sparse_state)

    macs, _ = thop_profile(
        copy.deepcopy(model),
        inputs=(x,),
        verbose=False
    )

    return {
        "latency_ms": latency,
        "dense_bytes": dense_bytes,
        "sparse_bytes": sparse_bytes,
        "macs": int(macs),
        "approx_flops": int(2 * macs)
    }


def actual_sparsity(model):

    weights = [
        m.weight.detach()
        for m in model.modules()
        if isinstance(m, (nn.Conv2d, nn.Linear))
    ]

    total = sum(w.numel() for w in weights)

    zeros = sum(
        (w == 0).sum().item()
        for w in weights
    )

    return zeros / total


def already_done(tag):

    if not METRICS.exists():
        return False

    df = pd.read_csv(METRICS)

    return (
        tag in df["tag"].values
        and (R / f"{tag}.pt").exists()
    )


def record(
    tag, method, model, td,
    target, seconds, energy,
    epochs
):

    acc = evaluate(model, td)
    measurements = measure(model)

    parameters = sum(
        p.numel() for p in model.parameters()
    )

    nonzero_parameters = sum(
        torch.count_nonzero(p).item()
        for p in model.parameters()
    )

    row = {
        "tag": tag,
        "model": method,
        "target_sparsity": target,
        "actual_weight_sparsity": actual_sparsity(model),
        "accuracy": acc,
        "parameters": parameters,
        "nonzero_parameters": nonzero_parameters,
        **measurements,
        "training_epochs": epochs,
        "training_seconds": seconds,
        "estimated_training_energy_kwh": energy,
        "device": str(DEVICE)
    }

    torch.save(
        model.state_dict(),
        R / f"{tag}.pt"
    )

    if METRICS.exists():

        df = pd.read_csv(METRICS)
        df = df[df["tag"] != tag]

        df = pd.concat(
            [df, pd.DataFrame([row])],
            ignore_index=True
        )

    else:
        df = pd.DataFrame([row])

    df.to_csv(METRICS, index=False)

    print("\nRESULT:", tag)
    print(f"Accuracy: {acc:.4f}")
    print(f"Parameters: {parameters:,}")
    print(f"Latency: {measurements['latency_ms']:.4f} ms")
    print(f"Dense size: {measurements['dense_bytes']:,} bytes")
    print(f"Sparse size: {measurements['sparse_bytes']:,} bytes")
    print(f"Estimated energy: {energy} kWh\n")


# ==================================================
# 5. BASELINE TEACHER
# ==================================================

def load_teacher():

    path = R / "teacher.pt"

    if not path.exists():
        raise FileNotFoundError(
            "Run the baseline study first."
        )

    teacher = Big().to(DEVICE)

    teacher.load_state_dict(
        torch.load(
            path,
            map_location=DEVICE,
            weights_only=True
        )
    )

    teacher.eval()

    return teacher


def baseline_study(dl, td):

    if already_done("teacher"):
        print("Teacher already completed.")
        return

    seed_all()

    teacher, seconds, energy = fit(
        Big(), dl, 5, "teacher"
    )

    record(
        "teacher", "teacher",
        teacher, td, 0,
        seconds, energy, 5
    )


# ==================================================
# 6. GLOBAL AND LAYER-WISE PRUNING
# ==================================================

def magnitude_study(dl, td, method):

    teacher = load_teacher()

    targets = [
        0, 0.25, 0.50, 0.75, 0.90, 0.95
    ]

    for sparsity in targets:

        tag = f"{method}_{int(sparsity * 100)}"

        if already_done(tag):
            print("Skipping completed:", tag)
            continue

        seed_all()

        model = copy.deepcopy(teacher)

        layers = [
            (m, "weight")
            for m in model.modules()
            if isinstance(m, (nn.Conv2d, nn.Linear))
        ]

        if method == "global":

            prune.global_unstructured(
                layers,
                pruning_method=prune.L1Unstructured,
                amount=sparsity
            )

        else:

            for module, _ in layers:

                prune.l1_unstructured(
                    module,
                    name="weight",
                    amount=sparsity
                )

        # Keep pruning masks fixed for all three epochs.
        model, seconds, energy = fit(
            model, dl, 3, tag
        )

        # Remove reparameterization only after training.
        for module, _ in layers:
            prune.remove(module, "weight")

        record(
            tag, method, model, td,
            sparsity, seconds, energy, 3
        )


# ==================================================
# 7. STRUCTURED FILTER REMOVAL
# ==================================================

def compact_model(teacher, fraction):

    conv1 = teacher.features[0]
    conv2 = teacher.features[3]

    # Rank filters using their L2 norms.
    norms1 = conv1.weight.detach().flatten(1).norm(
        p=2, dim=1
    )

    n1 = max(
        1,
        conv1.out_channels
        - round(conv1.out_channels * fraction)
    )

    keep1 = torch.topk(
        norms1, n1
    ).indices.sort().values

    # Rank second-layer filters after removing
    # input channels from the first layer.
    restricted = conv2.weight.detach()[:, keep1]

    norms2 = restricted.flatten(1).norm(
        p=2, dim=1
    )

    n2 = max(
        1,
        conv2.out_channels
        - round(conv2.out_channels * fraction)
    )

    keep2 = torch.topk(
        norms2, n2
    ).indices.sort().values

    compact = Big(n1, n2).to(DEVICE)

    with torch.no_grad():

        compact.features[0].weight.copy_(
            conv1.weight[keep1]
        )

        compact.features[0].bias.copy_(
            conv1.bias[keep1]
        )

        compact.features[3].weight.copy_(
            conv2.weight[keep2][:, keep1]
        )

        compact.features[3].bias.copy_(
            conv2.bias[keep2]
        )

        columns = (
            keep2[:, None] * 49
            + torch.arange(49, device=DEVICE)
        ).flatten()

        compact.head[1].weight.copy_(
            teacher.head[1].weight[:, columns]
        )

        compact.head[1].bias.copy_(
            teacher.head[1].bias
        )

        compact.head[3].load_state_dict(
            teacher.head[3].state_dict()
        )

    return compact


def structured_study(dl, td):

    teacher = load_teacher()

    for fraction in [0.10, 0.25, 0.50]:

        tag = f"structured_{int(fraction * 100)}"

        if already_done(tag):
            print("Skipping completed:", tag)
            continue

        seed_all()

        model = compact_model(
            teacher, fraction
        )

        model, seconds, energy = fit(
            model, dl, 3, tag
        )

        record(
            tag, "structured",
            model, td, fraction,
            seconds, energy, 3
        )


# ==================================================
# 8. PLAIN AND DISTILLED STUDENTS
# ==================================================

def student_study(dl, td):

    teacher = load_teacher()

    for tag in [
        "plain_student",
        "distilled_student"
    ]:

        if already_done(tag):
            print("Skipping completed:", tag)
            continue

        seed_all()

        model = Small()

        distillation_teacher = (
            teacher
            if tag == "distilled_student"
            else None
        )

        model, seconds, energy = fit(
            model,
            dl,
            5,
            tag,
            teacher=distillation_teacher
        )

        record(
            tag, tag, model, td,
            0, seconds, energy, 5
        )


# ==================================================
# 9. GRAPHS AND EFFICIENCY TABLE
# ==================================================

def save_plot(filename):

    plt.tight_layout()
    plt.savefig(R / filename, dpi=150)
    plt.close()


def generate_outputs():

    if not METRICS.exists():
        return

    df = pd.read_csv(METRICS)

    teacher = df[df["model"] == "teacher"]

    if teacher.empty:
        return

    baseline = teacher.iloc[0]

    # Accuracy vs actual unstructured sparsity.
    plt.figure(figsize=(8, 5))

    plotted = False

    for method in ["global", "layerwise"]:

        sub = df[
            df["model"] == method
        ].sort_values("actual_weight_sparsity")

        if not sub.empty:

            plt.plot(
                sub["actual_weight_sparsity"] * 100,
                sub["accuracy"] * 100,
                marker="o",
                label=method
            )

            plotted = True

    plt.axhline(
        baseline["accuracy"] * 100,
        linestyle="--",
        label="Teacher"
    )

    plt.xlabel("Actual weight sparsity (%)")
    plt.ylabel("Test accuracy (%)")
    plt.title("Magnitude pruning")

    if plotted:
        plt.legend()

    save_plot("sparsity_accuracy.png")

    # Structured pruning.
    sub = df[
        df["model"] == "structured"
    ].sort_values("target_sparsity")

    if not sub.empty:

        plt.figure(figsize=(7, 4))

        plt.plot(
            [0] + list(sub["target_sparsity"] * 100),
            [baseline["accuracy"] * 100]
            + list(sub["accuracy"] * 100),
            marker="o"
        )

        plt.xlabel("Target filter removal (%)")
        plt.ylabel("Test accuracy (%)")
        plt.title("Structured pruning")

        save_plot("structured_accuracy.png")

    # Latency comparison.
    plt.figure(figsize=(12, 6))

    plt.bar(df["tag"], df["latency_ms"])

    plt.xticks(rotation=60, ha="right")
    plt.ylabel("Median latency (ms)")
    plt.title("Measured inference latency")

    save_plot("latency_comparison.png")

    # Dense vs sparse serialized size.
    positions = np.arange(len(df))

    plt.figure(figsize=(12, 6))

    plt.bar(
        positions - 0.2,
        df["dense_bytes"] / 1024**2,
        width=0.4,
        label="Dense"
    )

    plt.bar(
        positions + 0.2,
        df["sparse_bytes"] / 1024**2,
        width=0.4,
        label="Sparse COO"
    )

    plt.xticks(
        positions,
        df["tag"],
        rotation=60,
        ha="right"
    )

    plt.ylabel("Serialized size (MiB)")
    plt.legend()
    plt.title("Dense vs sparse storage")

    save_plot("storage_comparison.png")

    # Plain vs distilled student.
    students = df[
        df["model"].isin([
            "plain_student",
            "distilled_student"
        ])
    ]

    if len(students) == 2:

        plt.figure(figsize=(6, 4))

        plt.bar(
            students["model"],
            students["accuracy"] * 100
        )

        plt.ylabel("Test accuracy (%)")
        plt.title("Knowledge distillation")

        save_plot("student_comparison.png")

    # Efficiency comparison table.
    efficiency = df.copy()

    efficiency["data_train_samples"] = 60000
    efficiency["data_test_samples"] = 10000

    efficiency["data_training_exposures"] = (
        efficiency["training_epochs"] * 60000
    )

    efficiency["parameter_reduction_pct"] = (
        1 - efficiency["parameters"]
        / baseline["parameters"]
    ) * 100

    efficiency["dense_size_reduction_pct"] = (
        1 - efficiency["dense_bytes"]
        / baseline["dense_bytes"]
    ) * 100

    efficiency["latency_speedup"] = (
        baseline["latency_ms"]
        / efficiency["latency_ms"]
    )

    efficiency["compute_reduction_pct"] = (
        1 - efficiency["macs"]
        / baseline["macs"]
    ) * 100

    columns = [
        "tag",
        "data_train_samples",
        "data_test_samples",
        "data_training_exposures",
        "accuracy",
        "parameters",
        "parameter_reduction_pct",
        "nonzero_parameters",
        "dense_bytes",
        "sparse_bytes",
        "dense_size_reduction_pct",
        "latency_ms",
        "latency_speedup",
        "macs",
        "approx_flops",
        "compute_reduction_pct",
        "training_seconds",
        "estimated_training_energy_kwh"
    ]

    efficiency[columns].to_csv(
        R / "efficiency_table.csv",
        index=False
    )

    generate_report(df, baseline)


# ==================================================
# 10. AUTOMATIC REPORT
# ==================================================

def generate_report(df, baseline):

    lines = [
        "# Lab 7: Compression",
        "",
        "## Objective",
        "",
        "This laboratory investigates pruning and "
        "knowledge distillation on Fashion-MNIST.",
        "",
        "## Experimental setup",
        "",
        f"- Device: {DEVICE}",
        "- Seed: 42",
        "- Training samples: 60,000",
        "- Test samples: 10,000",
        "- Batch size: 128",
        "- Optimizer: Adam",
        "- Learning rate: 0.001",
        "",
        "## Baseline CNN",
        "",
        f"- Parameters: {int(baseline['parameters']):,}",
        f"- Test accuracy: {baseline['accuracy']:.2%}",
        f"- Latency: {baseline['latency_ms']:.4f} ms",
        f"- Dense size: {int(baseline['dense_bytes']):,} bytes",
        "",
        "## Experimental results",
        "",
        "| Model | Accuracy | Parameters | Latency (ms) | "
        "Dense MiB | Sparse MiB | MACs | Energy (kWh) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|"
    ]

    for _, row in df.iterrows():

        energy = row["estimated_training_energy_kwh"]

        energy_text = (
            "Unavailable"
            if pd.isna(energy)
            else f"{energy:.6f}"
        )

        lines.append(
            f"| {row['tag']} "
            f"| {row['accuracy']:.2%} "
            f"| {int(row['parameters']):,} "
            f"| {row['latency_ms']:.4f} "
            f"| {row['dense_bytes']/1024**2:.3f} "
            f"| {row['sparse_bytes']/1024**2:.3f} "
            f"| {int(row['macs']):,} "
            f"| {energy_text} |"
        )

    lines.extend([
        "",
        "## Pruning and storage",
        "",
        "Global pruning ranks weights across all layers. "
        "Layer-wise pruning ranks weights independently "
        "within each layer.",
        "",
        "Fixed masks are maintained during three epochs "
        "of fine-tuning.",
        "",
        "Unstructured pruning creates zero weights "
        "without reducing the dimensions of dense tensors. "
        "Therefore, ordinary dense CPU kernels may still "
        "perform the same operations.",
        "",
        "Sparse COO storage includes index overhead and "
        "may be larger than dense storage at low sparsity.",
        "",
        "Structured pruning physically removes filters "
        "and adjusts the connected layers. This reduces "
        "model dimensions and nominal compute.",
        "",
        "Structured removal targets refer to filters, "
        "not the percentage of remaining zero weights.",
        "",
        "## Knowledge distillation",
        "",
        "Temperature = 4; alpha = 0.5.",
        "",
        "The plain and distilled students use the same "
        "architecture, initialization seed and five-epoch "
        "training budget.",
        "",
        "The implementation uses PyTorch KL divergence "
        "with teacher probabilities as the target, "
        "corresponding to KL(teacher || student).",
        ""
    ])

    plain = df[df["tag"] == "plain_student"]
    distilled = df[df["tag"] == "distilled_student"]

    if not plain.empty and not distilled.empty:

        p = plain.iloc[0]
        d = distilled.iloc[0]

        difference = (
            d["accuracy"] - p["accuracy"]
        ) * 100

        lines.extend([
            f"Plain student accuracy: {p['accuracy']:.2%}.",
            "",
            f"Distilled student accuracy: {d['accuracy']:.2%}.",
            "",
            f"Distillation accuracy difference: "
            f"{difference:+.2f} percentage points.",
            ""
        ])

    lines.extend([
        "## Efficiency pillars",
        "",
        "- Data: identical Fashion-MNIST datasets.",
        "- Model: parameter counts and serialized size.",
        "- Compute: nominal MACs and approximate FLOPs.",
        "- Energy: estimated training energy from CodeCarbon.",
        "",
        "Energy values describe the individual training "
        "or fine-tuning runs, not total lifecycle energy. "
        "Teacher training energy is excluded from student "
        "and pruning runs.",
        "",
        "CodeCarbon energy values are estimates, not "
        "direct power-meter measurements.",
        "",
        "Latency is measured with batch size 1 after "
        "30 warm-up iterations, using the median of "
        "100 timed iterations.",
        "",
        "Approximate FLOPs use two operations per MAC. "
        "Dense compute estimates do not assume sparse "
        "hardware acceleration.",
        "",
        "## Conclusion",
        "",
        "Compression involves trade-offs between "
        "accuracy, storage, latency, compute and energy. "
        "The final choice should depend on the measured "
        "results rather than sparsity alone.",
        "",
        "## References",
        "",
        "Vijay Janapa Reddi, Introduction to Machine "
        "Learning Systems: Principles and Practices "
        "of Engineering Artificially Intelligent Systems.",
        "",
        "Chapter 9: Efficient AI.",
        "",
        "Chapter 10: Sections 10.3 and 10.4.",
        ""
    ])

    (LAB / "report.md").write_text(
        "\n".join(lines),
        encoding="utf-8"
    )


# ==================================================
# 11. MAIN PROGRAM
# ==================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--study",
        choices=[
            "baseline",
            "global",
            "layerwise",
            "structured",
            "students",
            "report",
            "all"
        ],
        default="baseline"
    )

    args = parser.parse_args()

    seed_all()

    print("Starting Machine Learning Lab 7")
    print("Device:", DEVICE)
    print("Study:", args.study)

    versions = {
        "python": sys.version,
        "torch": torch.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "device": str(DEVICE)
    }

    (R / "versions.json").write_text(
        json.dumps(versions, indent=2),
        encoding="utf-8"
    )

    if args.study == "report":

        generate_outputs()
        print("Report generated.")
        return

    dl, td = make_loaders()

    if args.study in ["baseline", "all"]:
        baseline_study(dl, td)

    if args.study in ["global", "all"]:
        magnitude_study(dl, td, "global")

    if args.study in ["layerwise", "all"]:
        magnitude_study(dl, td, "layerwise")

    if args.study in ["structured", "all"]:
        structured_study(dl, td)

    if args.study in ["students", "all"]:
        student_study(dl, td)

    generate_outputs()

    print("\nLab 7 study completed.")
    print("Results:", R)
    print("Report:", LAB / "report.md")


if __name__ == "__main__":

    import multiprocessing

    multiprocessing.freeze_support()

    main()
