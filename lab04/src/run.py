
import sys
import json
import hashlib
import platform
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn import __version__ as sklearn_version
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    OneHotEncoder, StandardScaler
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.base import BaseEstimator, TransformerMixin


ROOT = Path(__file__).resolve().parents[2]
LAB = ROOT / "lab04"
RESULTS = LAB / "results"

RESULTS.mkdir(parents=True, exist_ok=True)

SEED = 42
np.random.seed(SEED)

NUMERIC = [
    "age", "fnlwgt", "education-num",
    "capital-gain", "capital-loss", "hours-per-week"
]

CATEGORICAL = [
    "workclass", "education", "marital-status",
    "occupation", "relationship", "race",
    "sex", "native-country"
]

RANGES = {
    "age": (17, 90),
    "hours-per-week": (1, 99),
    "capital-gain": (0, 99999),
    "capital-loss": (0, 4356)
}

TARGET = "income"


def save_json(filename, data):
    path = RESULTS / filename
    path.write_text(
        json.dumps(data, indent=2, default=str),
        encoding="utf-8"
    )


def sha256_file(path):
    digest = hashlib.sha256()

    with open(path, "rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()



class NumericClipper(BaseEstimator, TransformerMixin):
    """Learn clipping boundaries from training data only."""

    def __init__(self, lower=0.01, upper=0.99):
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):
        frame = pd.DataFrame(X).apply(
            pd.to_numeric, errors="coerce"
        )
        self.lower_bounds_ = frame.quantile(
            self.lower
        ).to_numpy()
        self.upper_bounds_ = frame.quantile(
            self.upper
        ).to_numpy()
        return self

    def transform(self, X):
        values = pd.DataFrame(X).apply(
            pd.to_numeric, errors="coerce"
        ).to_numpy(dtype=float)

        return np.clip(
            values,
            self.lower_bounds_,
            self.upper_bounds_
        )


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Create features without learning from test data."""

    def __init__(self, enabled=True):
        self.enabled = enabled

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        frame = X.copy()

        if self.enabled:
            gain = pd.to_numeric(
                frame["capital-gain"], errors="coerce"
            )
            loss = pd.to_numeric(
                frame["capital-loss"], errors="coerce"
            )
            hours = pd.to_numeric(
                frame["hours-per-week"], errors="coerce"
            )
            age = pd.to_numeric(
                frame["age"], errors="coerce"
            )

            frame["capital_net"] = gain - loss

            denominator = (age - 16).clip(lower=1)

            frame["hours_per_age"] = (
                hours / denominator
            )

        return frame


class MissingMarkerCleaner(BaseEstimator, TransformerMixin):
    """Convert Adult's '?' markers to missing values."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        frame = X.copy()

        for col in frame.columns:
            frame[col] = frame[col].astype(object)
            frame[col] = frame[col].replace(
                ["?", "", "nan", "None"], np.nan
            )

        return frame


def build_pipeline(engineered):
    numeric_features = NUMERIC.copy()

    if engineered:
        numeric_features += [
            "capital_net", "hours_per_age"
        ]

    numeric_pipeline = Pipeline([
        ("clip", NumericClipper()),
        ("imputer", SimpleImputer(
            strategy="median"
        )),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(
            strategy="most_frequent"
        )),
        ("onehot", OneHotEncoder(
            handle_unknown="ignore"
        ))
    ])

    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, CATEGORICAL)
    ])

    return Pipeline([
        ("clean", MissingMarkerCleaner()),
        ("features", FeatureEngineer(
            enabled=engineered
        )),
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(
            max_iter=1000,
            random_state=SEED
        ))
    ])


print("\nLAB 4: DATA PIPELINE AND QUALITY")
print("=" * 55)

print("\n[1/8] Downloading UCI Adult dataset...")

dataset = fetch_openml(
    name="adult",
    version=2,
    as_frame=True
)

X = dataset.data.copy()

y = (
    dataset.target.astype(str)
    .str.strip()
    .str.rstrip(".")
)

raw = X.copy()
raw[TARGET] = y

raw_path = RESULTS / "raw_snapshot.csv"
raw.to_csv(raw_path, index=False)

raw_hash = sha256_file(raw_path)

print("Rows:", len(raw))
print("Columns:", len(raw.columns))
print("Raw SHA256:", raw_hash)


print("\n[2/8] Validating dataset...")

validation = {
    "dataset": "UCI Adult / OpenML version 2",
    "expected_rows": 48842,
    "actual_rows": int(len(raw)),
    "expected_columns": 15,
    "actual_columns": int(len(raw.columns)),
    "row_count_valid": len(raw) == 48842,
    "column_count_valid": len(raw.columns) == 15,
    "duplicate_rows": int(raw.duplicated().sum()),
    "schema": {},
    "range_violations": {},
    "missing_values": {},
    "target_counts": y.value_counts().to_dict()
}

for col in NUMERIC:
    values = pd.to_numeric(
        raw[col], errors="coerce"
    )

    nonmissing = raw[col].notna()

    validation["schema"][col] = {
        "expected_type": "integer",
        "non_numeric_values": int(
            (values.isna() & nonmissing).sum()
        ),
        "non_integer_values": int(
            ((values.dropna() % 1) != 0).sum()
        )
    }

for col in CATEGORICAL + [TARGET]:
    validation["schema"][col] = {
        "expected_type": "category",
        "observed_unique_values": int(
            raw[col].nunique(dropna=True)
        )
    }

for col, (lower, upper) in RANGES.items():
    values = pd.to_numeric(
        raw[col], errors="coerce"
    )

    violations = (
        values.notna()
        & ~values.between(lower, upper)
    )

    validation["range_violations"][col] = {
        "minimum_allowed": lower,
        "maximum_allowed": upper,
        "violation_count": int(violations.sum())
    }

for col in raw.columns:
    series = raw[col].astype("string").str.strip()

    marker_count = int(
        series.eq("?").fillna(False).sum()
    )

    validation["missing_values"][col] = {
        "null_count": int(raw[col].isna().sum()),
        "question_mark_count": marker_count
    }

validation["invalid_target_count"] = int(
    (~y.isin(["<=50K", ">50K"])).sum()
)

save_json("validation.json", validation)


print("\n[3/8] Checking label quality...")


grouped = raw.groupby(
    list(X.columns),
    dropna=False,
    observed=True
)[TARGET].nunique(dropna=False)

conflicting_groups = int(
    (grouped > 1).sum()
)

validation["conflicting_feature_groups"] = (
    conflicting_groups
)


audit_sample = raw.sample(
    n=50,
    random_state=SEED
).copy()

audit_sample.insert(
    0, "original_row_index", audit_sample.index
)

audit_sample["automated_range_check"] = "PASS"

for col, (lower, upper) in RANGES.items():
    values = pd.to_numeric(
        audit_sample[col], errors="coerce"
    )

    invalid = (
        values.notna()
        & ~values.between(lower, upper)
    )

    audit_sample.loc[
        invalid, "automated_range_check"
    ] = "FAIL"

audit_sample["manual_dictionary_review"] = (
    "PENDING"
)

audit_sample["review_notes"] = ""

audit_sample.to_csv(
    RESULTS / "label_audit_sample.csv",
    index=False
)

validation["label_audit"] = {
    "sample_size": 50,
    "random_seed": SEED,
    "conflicting_feature_groups": conflicting_groups,
    "manual_review_status": "PENDING",
    "note": (
        "Review the 50 sampled rows against the "
        "Adult data dictionary and record findings."
    )
}

save_json("validation.json", validation)



print("\n[4/8] Splitting train and test data...")

X_train, X_test, y_train, y_test = (
    train_test_split(
        X, y,
        test_size=0.30,
        stratify=y,
        random_state=SEED
    )
)

train_frame = X_train.copy()
train_frame[TARGET] = y_train

before_duplicates = len(train_frame)


train_frame = train_frame.drop_duplicates()

removed_duplicates = (
    before_duplicates - len(train_frame)
)

y_train = train_frame.pop(TARGET)
X_train = train_frame


split_info = {
    "seed": SEED,
    "test_fraction": 0.30,
    "stratified": True,
    "training_rows_before_deduplication":
        before_duplicates,
    "training_duplicates_removed":
        removed_duplicates,
    "training_rows_after_deduplication":
        len(X_train),
    "test_rows": len(X_test),
    "test_modified": False
}

save_json("split_info.json", split_info)

print("Training rows:", len(X_train))
print("Test rows:", len(X_test))
print("Training duplicates removed:",
      removed_duplicates)


print("\n[5/8] Training comparison models...")

comparison = []
fitted_models = {}

for engineered in [False, True]:
    label = (
        "With engineered features"
        if engineered else
        "Without engineered features"
    )

    print("\nTraining:", label)

    pipeline = build_pipeline(engineered)

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=FutureWarning
        )
        pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)

    accuracy = accuracy_score(
        y_test, predictions
    )

    f1 = f1_score(
        y_test, predictions,
        pos_label=">50K"
    )

    comparison.append({
        "model": label,
        "engineered": engineered,
        "accuracy": float(accuracy),
        "f1": float(f1)
    })

    fitted_models[engineered] = pipeline

    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 score: {f1:.4f}")

comparison_df = pd.DataFrame(comparison)

comparison_df.to_csv(
    RESULTS / "feature_comparison.csv",
    index=False
)


joblib.dump(
    fitted_models[True],
    RESULTS / "preprocessor.joblib"
)



print("\n[6/8] Hashing processed training data...")

final_pipeline = fitted_models[True]

processed_training = (
    final_pipeline.named_steps["preprocessor"]
    .transform(
        final_pipeline.named_steps["features"]
        .transform(
            final_pipeline.named_steps["clean"]
            .transform(X_train)
        )
    )
)


from scipy import sparse

digest = hashlib.sha256()

if sparse.issparse(processed_training):
    matrix = processed_training.tocsr()
    matrix.sort_indices()

    digest.update(b"csr")
    digest.update(
        np.asarray(matrix.shape, dtype="<i8").tobytes()
    )
    digest.update(
        np.asarray(matrix.data, dtype="<f8").tobytes()
    )
    digest.update(
        np.asarray(matrix.indices, dtype="<i8").tobytes()
    )
    digest.update(
        np.asarray(matrix.indptr, dtype="<i8").tobytes()
    )
else:
    matrix = np.asarray(
        processed_training,
        dtype="<f8"
    )

    digest.update(b"dense")
    digest.update(
        np.asarray(matrix.shape, dtype="<i8").tobytes()
    )
    digest.update(matrix.tobytes())

processed_hash = digest.hexdigest()

(RESULTS / "processed_sha256.txt").write_text(
    processed_hash + "\n",
    encoding="utf-8"
)

print("Processed SHA256:", processed_hash)



print("\n[7/8] Generating results and configuration...")

fig, ax = plt.subplots(figsize=(8, 5))

positions = np.arange(2)
width = 0.35

ax.bar(
    positions - width / 2,
    comparison_df["accuracy"],
    width,
    label="Accuracy"
)

ax.bar(
    positions + width / 2,
    comparison_df["f1"],
    width,
    label="F1"
)

ax.set_xticks(positions)
ax.set_xticklabels([
    "Baseline",
    "Engineered"
])

ax.set_ylim(0, 1)
ax.set_ylabel("Score")
ax.set_title(
    "Adult Dataset: Feature Engineering Comparison"
)
ax.legend()

plt.tight_layout()
plt.savefig(
    RESULTS / "feature_comparison.png",
    dpi=150
)
plt.close()

versions = {
    "python": sys.version.split()[0],
    "platform": platform.platform(),
    "numpy": np.__version__,
    "pandas": pd.__version__,
    "scikit_learn": sklearn_version,
    "joblib": joblib.__version__,
    "matplotlib": matplotlib.__version__,
    "pyyaml": yaml.__version__
}

save_json("versions.json", versions)

config = {
    "dataset": "OpenML Adult",
    "dataset_version": 2,
    "seed": SEED,
    "train_fraction": 0.70,
    "test_fraction": 0.30,
    "stratified_split": True,
    "numeric_imputation": "median",
    "categorical_imputation": "most_frequent",
    "clip_lower_quantile": 0.01,
    "clip_upper_quantile": 0.99,
    "duplicate_policy": "training only",
    "features": [
        "capital_net",
        "hours_per_age"
    ],
    "model": "LogisticRegression",
    "max_iter": 1000,
    "raw_sha256": raw_hash,
    "processed_sha256": processed_hash,
    "versions": versions
}

(RESULTS / "config.yaml").write_text(
    yaml.safe_dump(
        config,
        sort_keys=False
    ),
    encoding="utf-8"
)



print("\n[8/8] Writing report and data card...")

baseline = comparison[0]
engineered = comparison[1]

accuracy_change = (
    engineered["accuracy"] - baseline["accuracy"]
)

f1_change = (
    engineered["f1"] - baseline["f1"]
)

missing_lines = "\n".join(
    f"- {col}: "
    f"{info['null_count']} null values, "
    f"{info['question_mark_count']} '?' markers"
    for col, info in validation["missing_values"].items()
)

range_lines = "\n".join(
    f"- {col}: {info['violation_count']} violations "
    f"(allowed {info['minimum_allowed']}–"
    f"{info['maximum_allowed']})"
    for col, info in validation["range_violations"].items()
)

validation_report = f"""# Lab 4 Validation Report

## Dataset

Source: UCI Adult through OpenML version 2.

Rows: {len(raw)}
Columns: {len(raw.columns)}

Expected rows: 48,842
Expected columns: 15

Row count valid: {validation['row_count_valid']}
Column count valid: {validation['column_count_valid']}

## Schema

Numerical integer fields:
{", ".join(NUMERIC)}

Categorical fields:
{", ".join(CATEGORICAL + [TARGET])}

Detailed schema results are stored in validation.json.

## Range Validation

{range_lines}

## Missing Values

{missing_lines}

## Duplicates

Exact duplicate rows in raw dataset:
{validation['duplicate_rows']}

Duplicates removed from training:
{removed_duplicates}

Test rows were not deduplicated or removed.

## Label Quality

Feature groups with conflicting income labels:
{conflicting_groups}

A reproducible 50-row sample was saved as
label_audit_sample.csv.

Automated range checks are included.

Manual comparison with the Adult data dictionary
is still required. The manual review has not
been marked complete.

## Leakage Prevention

The dataset was split before cleaning.

Imputation, clipping, scaling and category encoding
were fitted on the training data only.

The test data was transformed using the fitted
training pipeline.

## Hashes

Raw snapshot SHA256:
{raw_hash}

Processed training matrix SHA256:
{processed_hash}

See config.yaml for the experiment settings.
"""

(RESULTS / "validation_report.md").write_text(
    validation_report,
    encoding="utf-8"
)

data_card = f"""# Lab 4 Data Card: UCI Adult

## 1. Dataset Overview

Dataset: Adult (Census Income)
Source: UCI Machine Learning Repository,
accessed through OpenML version 2.

Original dataset size: {len(raw):,} rows
Total columns: {len(raw.columns)}
Input features: {len(X.columns)}
Target: income

## 2. Problem Definition

The objective is to predict whether an individual's
annual income is above $50,000 using demographic,
education and employment-related information.

The input contains 14 features, including numerical
values such as age and working hours, and categorical
values such as occupation and education.

The output is a binary income category:
<=50K or >50K.

Accuracy and F1 score are used to evaluate the model.
F1 is important because the income categories are
not equally represented.

The main engineering constraints are preventing
data leakage and making the pipeline reproducible.
Training and testing data must be separated before
cleaning, and preprocessing must learn its parameters
only from training data.

The experiment also examines whether two engineered
features improve the performance of a logistic
regression baseline.

## 3. Source and Collection

The dataset is retrieved using:

fetch_openml(name="adult", version=2, as_frame=True)

The data originates from US Census information
and is commonly used for classification research.

## 4. License

The dataset is distributed through UCI and OpenML.

The applicable license and reuse conditions should
be checked on the dataset's current source page
before redistribution or commercial use.

No license has been independently verified
by this script.

## 5. Data Structure

Numerical columns:
{", ".join(NUMERIC)}

Categorical columns:
{", ".join(CATEGORICAL)}

Target:
income

## 6. Cleaning and Processing

The dataset is split into training and test sets
using a 70/30 stratified split.

Exact duplicates are removed from training only.

Missing numerical values are replaced using
training-set medians.

Missing categorical values are replaced using
training-set modes.

Numerical outliers are clipped using training-set
1st and 99th percentiles.

Numerical features are standardized.

Categorical features are one-hot encoded.

The fitted pipeline is saved as
results/preprocessor.joblib.

## 7. Engineered Features

capital_net = capital-gain - capital-loss

hours_per_age = hours-per-week / (age - 16)

These features are computed without fitting
statistics on the test set.

## 8. Label Quality

Conflicting feature groups:
{conflicting_groups}

A random sample of 50 rows is stored in
results/label_audit_sample.csv.

Manual data-dictionary review is pending.

## 9. Known Biases

The data contains demographic attributes including
race and sex, and reflects historical patterns in
income and employment.

These patterns may contain social inequalities
and should not be interpreted as fair or neutral.

A model trained on this dataset may reproduce
biases present in the historical data.

## 10. Privacy Considerations

The dataset contains personal demographic and
employment-related attributes.

Even though it is publicly distributed, the
information is sensitive in nature.

Predictions should not be used to make important
decisions about individuals without additional
privacy, fairness and legal review.

## 11. Model Evaluation

Baseline accuracy: {baseline['accuracy']:.4f}
Baseline F1: {baseline['f1']:.4f}

Engineered accuracy: {engineered['accuracy']:.4f}
Engineered F1: {engineered['f1']:.4f}

Accuracy change: {accuracy_change:+.4f}
F1 change: {f1_change:+.4f}

The comparison shows whether the two derived
features helped or reduced performance in
this particular experiment.

## 12. Reproducibility

Random seed: {SEED}
Training fraction: 70%
Test fraction: 30%

Raw snapshot SHA256:
{raw_hash}

Processed training SHA256:
{processed_hash}

Configuration:
results/config.yaml

Environment versions:
results/versions.json

To reproduce from the repository root:

python lab04/src/run.py

The command requires the specified Python packages
and access to the OpenML dataset.

## 13. Limitations

The dataset represents historical information,
so its relationships may not reflect current
income patterns.

The experiment uses logistic regression rather
than a wider range of classification models.

Accuracy and F1 do not establish fairness.

The dataset license must be verified separately,
and the 50-row manual audit must be completed
before claiming the full data review is finished.
"""

(LAB / "data_card.md").write_text(
    data_card,
    encoding="utf-8"
)

report = f"""# Lab 4: Data Pipeline and Quality

## Goal

Build a reproducible and leakage-free data pipeline
for the UCI Adult dataset.

## Method

The dataset contains {len(raw):,} rows and
{len(raw.columns)} columns.

A stratified 70/30 train/test split was performed
before cleaning.

Missing values were handled using training-only
imputation, numerical outliers were clipped using
training-only percentiles, and exact duplicates
were removed from the training data only.

Two features were engineered:
capital_net and hours_per_age.

Logistic regression was evaluated with and without
the engineered features.

## Results

| Model | Accuracy | F1 |
|---|---:|---:|
| Baseline | {baseline['accuracy']:.4f} | {baseline['f1']:.4f} |
| Engineered | {engineered['accuracy']:.4f} | {engineered['f1']:.4f} |

Accuracy change: {accuracy_change:+.4f}

F1 change: {f1_change:+.4f}

## Validation

Raw duplicate rows: {validation['duplicate_rows']}

Training duplicates removed: {removed_duplicates}

Conflicting feature groups: {conflicting_groups}

Detailed results:
results/validation_report.md

## Leakage Prevention

The test set was not used to fit imputers,
clipping thresholds, scaling parameters or
categorical encoders.

The fitted preprocessing and classification
pipeline was saved using joblib.

## Dataset Hash

{processed_hash}

## Data Documentation

See data_card.md for dataset source, structure,
processing, known biases, privacy considerations
and reproducibility information.

## Remaining Manual Checks

Review all 50 sampled rows against the Adult
data dictionary and record the findings.

Verify dataset license terms on the source page.

## Conclusion

This laboratory demonstrated how data quality,
feature engineering and reproducibility affect
a machine learning workflow.

The feature comparison provides measured evidence
of whether the two engineered features helped
the logistic regression model.

The validation checks and saved configuration
make the experiment easier to inspect and repeat.
"""

(LAB / "report.md").write_text(
    report,
    encoding="utf-8"
)

print("\n" + "=" * 55)
print("LAB 4 EXPERIMENT COMPLETED")
print("=" * 55)

print("\nBaseline:")
print(
    f"Accuracy: {baseline['accuracy']:.4f} | "
    f"F1: {baseline['f1']:.4f}"
)

print("\nEngineered:")
print(
    f"Accuracy: {engineered['accuracy']:.4f} | "
    f"F1: {engineered['f1']:.4f}"
)

print("\nResults saved in:")
print(RESULTS)

print("\nReport:")
print(LAB / "report.md")

print("\nData card:")
print(LAB / "data_card.md")

print("\nIMPORTANT:")
print("Complete the 50-row manual label audit.")
print("Verify the dataset license before submission.")
