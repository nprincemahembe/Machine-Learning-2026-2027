# Lab 4 Data Card: UCI Adult

## 1. Dataset Overview

Dataset: Adult (Census Income)
Source: UCI Machine Learning Repository,
accessed through OpenML version 2.

Original dataset size: 48,842 rows
Total columns: 15
Input features: 14
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
age, fnlwgt, education-num, capital-gain, capital-loss, hours-per-week

Categorical columns:
workclass, education, marital-status, occupation, relationship, race, sex, native-country

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
5

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

Baseline accuracy: 0.8540
Baseline F1: 0.6601

Engineered accuracy: 0.8578
Engineered F1: 0.6696

Accuracy change: +0.0038
F1 change: +0.0095

The comparison shows whether the two derived
features helped or reduced performance in
this particular experiment.

## 12. Reproducibility

Random seed: 42
Training fraction: 70%
Test fraction: 30%

Raw snapshot SHA256:
59959a880c002fba75a447b534e6e9d0bb6151708f6154f8f13ccdd552335a34

Processed training SHA256:
d40d0650fc6f5e759dc9a9ee44c99bd9184b5b42757a7884dc97be0668699ced

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
