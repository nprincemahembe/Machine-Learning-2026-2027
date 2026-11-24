# Lab 4: Data Pipeline and Quality

## Goal

Build a reproducible and leakage-free data pipeline
for the UCI Adult dataset.

## Method

The dataset contains 48,842 rows and
15 columns.

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
| Baseline | 0.8540 | 0.6601 |
| Engineered | 0.8578 | 0.6696 |

Accuracy change: +0.0038

F1 change: +0.0095

## Validation

Raw duplicate rows: 52

Training duplicates removed: 24

Conflicting feature groups: 5

Detailed results:
results/validation_report.md

## Leakage Prevention

The test set was not used to fit imputers,
clipping thresholds, scaling parameters or
categorical encoders.

The fitted preprocessing and classification
pipeline was saved using joblib.

## Dataset Hash

d40d0650fc6f5e759dc9a9ee44c99bd9184b5b42757a7884dc97be0668699ced

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
