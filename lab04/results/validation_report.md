# Lab 4 Validation Report

## Dataset

Source: UCI Adult through OpenML version 2.

Rows: 48842
Columns: 15

Expected rows: 48,842
Expected columns: 15

Row count valid: True
Column count valid: True

## Schema

Numerical integer fields:
age, fnlwgt, education-num, capital-gain, capital-loss, hours-per-week

Categorical fields:
workclass, education, marital-status, occupation, relationship, race, sex, native-country, income

Detailed schema results are stored in validation.json.

## Range Validation

- age: 0 violations (allowed 17–90)
- hours-per-week: 0 violations (allowed 1–99)
- capital-gain: 0 violations (allowed 0–99999)
- capital-loss: 0 violations (allowed 0–4356)

## Missing Values

- age: 0 null values, 0 '?' markers
- workclass: 2799 null values, 0 '?' markers
- fnlwgt: 0 null values, 0 '?' markers
- education: 0 null values, 0 '?' markers
- education-num: 0 null values, 0 '?' markers
- marital-status: 0 null values, 0 '?' markers
- occupation: 2809 null values, 0 '?' markers
- relationship: 0 null values, 0 '?' markers
- race: 0 null values, 0 '?' markers
- sex: 0 null values, 0 '?' markers
- capital-gain: 0 null values, 0 '?' markers
- capital-loss: 0 null values, 0 '?' markers
- hours-per-week: 0 null values, 0 '?' markers
- native-country: 857 null values, 0 '?' markers
- income: 0 null values, 0 '?' markers

## Duplicates

Exact duplicate rows in raw dataset:
52

Duplicates removed from training:
24

Test rows were not deduplicated or removed.

## Label Quality

Feature groups with conflicting income labels:
5

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
59959a880c002fba75a447b534e6e9d0bb6151708f6154f8f13ccdd552335a34

Processed training matrix SHA256:
d40d0650fc6f5e759dc9a9ee44c99bd9184b5b42757a7884dc97be0668699ced

See config.yaml for the experiment settings.
