# Lab 5: Frameworks and Automatic Differentiation

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
0

Maximum difference compared with numerical gradients:
2.80249157214e-11

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
- Test accuracy: 0.9100

PyTorch:
- Training observations: 800
- Epochs: 150
- Test accuracy: 0.8800

The two models use the same architecture but different
training schedules. Their accuracies are not a controlled
comparison of framework quality.

## 5. Training Step Speed

Manual autograd median:
0.437600 ms

PyTorch median:
0.839100 ms

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
0.068800 ms

Traced median:
0.056900 ms

Tracing was faster in this experiment. It may reduce Python execution overhead by representing operations in a reusable graph.

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

Tensor shape: [512, 512]
Tensor dtype: torch.float32

Non-contiguous strides:
[1, 512]

Contiguous strides:
[512, 1]

Non-contiguous matrix multiplication:
3.409600 ms

Contiguous matrix multiplication:
3.443200 ms

The non-contiguous view was not slower in this test. Performance depends on the operation, memory layout and underlying matrix multiplication implementation.

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

Random seed: 42
PyTorch version: 2.3.1+cpu
Python version: 3.11.9
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
