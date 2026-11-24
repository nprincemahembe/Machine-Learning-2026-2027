# Lab 2: Neural Network From Scratch

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

Initial loss: 2.302557

Expected initial loss: 2.302585

## Gradient Checking

Numerical gradients were calculated using
the central difference method.

Epsilon: 1e-5

Relative error threshold: 1e-6

Gradient checks passed: 10/10

Required: at least 9/10.

## Training Results

| Width | Learning Rate | Test Accuracy | Final Loss |
|---|---|---|---|
| 16 | 0.001 | 0.1019 | 2.302549 |
| 16 | 0.01 | 0.1019 | 2.302475 |
| 16 | 0.1 | 0.6722 | 0.899199 |
| 64 | 0.001 | 0.1019 | 2.302516 |
| 64 | 0.01 | 0.1019 | 2.302259 |
| 64 | 0.1 | 0.9444 | 0.143165 |

The loss curves are saved in:

results/loss_curves.png

## Timing Results

Each operation was warmed up once and
measured 100 times.

The median timing is reported in milliseconds.

| Width | Forward ms | Backward ms |
|---|---|---|
| 16 | 0.108250 | 0.073000 |
| 64 | 0.092950 | 0.123000 |

For width 16, forward propagation took longer in the measured runs. For width 64, backward propagation took longer in the measured runs.

Forward propagation calculates the outputs
using matrix multiplication.

Backward propagation calculates gradients
through the chain rule and additional
matrix multiplications.

The measured difference depends on the
network width and the computer.

## Three Conclusions

1. The highest test accuracy was
0.9444, achieved with
width 64 and learning rate
0.1.

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
