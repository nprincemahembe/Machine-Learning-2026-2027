
import csv
import warnings
from pathlib import Path

import torch
import torch.nn as nn
from torchinfo import summary
from thop import profile

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "lab03" / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

torch.manual_seed(42)


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
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


class AttentionModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.patch_projection = nn.Linear(49, 64)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=64,
            nhead=4,
            dim_feedforward=128,
            batch_first=True
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=2,
            enable_nested_tensor=False
        )

        self.classifier = nn.Linear(64, 10)

    def forward(self, x):
        batch_size = x.size(0)

        patches = x.unfold(2, 7, 7).unfold(3, 7, 7)
        patches = patches.contiguous().reshape(
            batch_size, 16, 49
        )

        x = self.patch_projection(patches)
        x = self.encoder(x)
        x = x.mean(dim=1)

        return self.classifier(x)


def linear_flops(inputs, outputs, tokens=1):
    return 2 * inputs * outputs * tokens


def convolution_flops(k, cin, cout, h, w):
    return 2 * k * k * cin * cout * h * w


def attention_encoder_flops():
    seq = 16
    dim = 64
    ff = 128

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


manual_flops = {
    "MLP": (
        linear_flops(784, 256)
        + linear_flops(256, 128)
        + linear_flops(128, 10)
    ),

    "CNN": (
        convolution_flops(3, 1, 16, 28, 28)
        + convolution_flops(3, 16, 32, 14, 14)
        + linear_flops(32 * 7 * 7, 10)
    ),

    "Attention": (
        linear_flops(49, 64, 16)
        + 2 * attention_encoder_flops()
        + linear_flops(64, 10)
    )
}


models = {
    "MLP": MLP(),
    "CNN": CNN(),
    "Attention": AttentionModel()
}

rows = []

print("\nLAB 3: FLOPs COMPARISON")
print("=" * 65)

for name, model in models.items():
    model.eval()

    dummy_input = torch.randn(1, 1, 28, 28)

    model_info = summary(
        model,
        input_size=(1, 1, 28, 28),
        device="cpu",
        verbose=0
    )

    parameters = int(model_info.total_params)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        macs, _ = profile(
            model,
            inputs=(dummy_input,),
            verbose=False
        )

    macs = int(macs)

    
    estimated_flops = 2 * macs

    hand = manual_flops[name]

    difference = estimated_flops - hand
    difference_percent = 100 * difference / hand

    if name == "Attention":
        note = (
            "THOP may omit internal attention operations; "
            "its result is not a complete FLOP count."
        )
    else:
        note = (
            "Differences may reflect pooling, activations, "
            "bias and profiler counting conventions."
        )

    row = {
        "model": name,
        "parameters": parameters,
        "hand_calculated_flops": hand,
        "thop_macs": macs,
        "thop_estimated_flops": estimated_flops,
        "difference_flops": difference,
        "difference_percent": round(
            difference_percent, 2
        ),
        "notes": note
    }

    rows.append(row)

    print(f"\n{name}")
    print(f"Parameters: {parameters:,}")
    print(f"Hand-calculated FLOPs: {hand:,}")
    print(f"THOP MACs: {macs:,}")
    print(f"THOP estimated FLOPs: {estimated_flops:,}")
    print(f"Difference: {difference_percent:.2f}%")


output_path = RESULTS / "flops_comparison.csv"

with output_path.open(
    "w", newline="", encoding="utf-8"
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=list(rows[0].keys())
    )

    writer.writeheader()
    writer.writerows(rows)


print("\n===================================")
print("FLOPs COMPARISON COMPLETED")
print("===================================")

print("\nSaved to:")
print(output_path)
