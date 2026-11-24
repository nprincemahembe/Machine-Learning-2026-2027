# Lab 3: Architectures and Their Costs

## 1. Goal

The aim of this laboratory was to implement and compare three deep neural network architectures using PyTorch: a Multi-Layer Perceptron (MLP), a Convolutional Neural Network (CNN), and an attention-based model.

The models were compared based on accuracy, number of parameters, computational cost, inference latency, and memory-access patterns.

## 2. Dataset and Method

The Fashion-MNIST dataset was used to train and evaluate the models. The dataset contains images of clothing items belonging to 10 different classes.

The experiments used a random seed of 42 for reproducibility. All three models were trained using the Adam optimizer for five epochs, with a learning rate of 0.001 and a batch size of 128.

The models were evaluated using the test dataset, which was not used for training.

## 3. Model Architectures

**Multi-Layer Perceptron (MLP)**

The MLP takes a flattened image containing 784 pixels and processes it through two hidden layers containing 256 and 128 neurons. ReLU activation is applied after each hidden layer, followed by an output layer containing 10 neurons.

**Convolutional Neural Network (CNN)**

The CNN uses two convolutional layers with 16 and 32 output channels. Each convolution is followed by ReLU activation and max pooling. The extracted features are flattened and passed through a fully connected classification layer.

**Attention Model**

The attention model divides each image into 16 patches of size 7 × 7 pixels. Each patch is projected into 64 features and processed by two Transformer encoder layers. The model uses four attention heads, a feed-forward dimension of 128, mean pooling, and a final classification layer.

## 4. Parameter Counts

| Model | Parameters |
|---|---:|
| MLP | 235,146 |
| CNN | 20,490 |
| Attention | 70,794 |

The MLP has the highest number of parameters because its fully connected layers require separate weights between neurons.

The CNN has the fewest parameters because convolutional filters are shared across different parts of an image.

The attention model falls between the other two models in parameter count.

The laboratory manual specifies a target of approximately 100,000–150,000 parameters. However, the prescribed architectures do not all meet this target. The experiments retained the specified architectures instead of changing their structure.

## 5. Hand-Calculated FLOPs

FLOPs represent the approximate number of floating-point operations required to perform computations.

For a linear layer:

FLOPs = 2 × input features × output features

For the first MLP layer:

2 × 784 × 256 = **401,408 FLOPs**

For a convolutional layer:

FLOPs = 2 × kernel height × kernel width × input channels × output channels × output height × output width

For the first CNN convolution:

2 × 3 × 3 × 1 × 16 × 28 × 28 = **225,792 FLOPs**

For the attention model, the query, key, and value projections require:

2 × 16 × 64 × 3 × 64 = **393,216 FLOPs**

The attention score calculation requires:

2 × 16² × 64 = **32,768 FLOPs**

These calculations represent selected layers rather than the total computational cost of each model.

## 6. Hand Calculations Compared With Profiling

I compared the hand-calculated FLOPs with measurements obtained using THOP. Since THOP reports multiply-accumulate operations (MACs), I multiplied the measured MACs by two to estimate FLOPs.

| Model | Hand-calculated FLOPs | THOP MACs | THOP estimated FLOPs | Difference |
|---|---:|---:|---:|---:|
| MLP | 469,504 | 234,752 | 469,504 | 0.00% |
| CNN | 2,063,488 | 1,031,744 | 2,063,488 | 0.00% |
| Attention | 2,329,856 | 50,816 | 101,632 | -95.64% |

For the MLP, the hand calculation and THOP estimate matched exactly at 469,504 FLOPs. The CNN also produced matching results of 2,063,488 FLOPs.

This shows that THOP successfully counted the main linear and convolution operations included in my manual calculations.

However, the attention model showed a difference of 95.64%. My hand calculation estimated 2,329,856 FLOPs, while THOP reported only 101,632 FLOPs.

This difference occurs because THOP does not fully count the internal operations of the Transformer encoder, including some attention-related matrix calculations.

The result does not mean that the attention model actually requires fewer operations. It shows that profiling tools have limitations, especially when working with more complex architectures.

I also learned that FLOPs alone cannot explain the actual inference speed of a model. Memory access, intermediate calculations, and the way operations are implemented can affect performance.

The full measurements are saved in `results/flops_comparison.csv`.

## 7. Test Accuracy

| Model | Test accuracy |
|---|---:|
| MLP | 87.94% |
| CNN | 89.51% |
| Attention | 83.98% |

The CNN achieved the highest test accuracy at 89.51%.

The MLP followed with 87.94%, while the attention model achieved 83.98%.

The CNN performed well because convolutional filters can identify local patterns in images, such as edges and shapes.

The attention model had the lowest accuracy in this experiment. This does not mean that attention is always less effective, but it suggests that the CNN was better suited to this particular task and training setup.

## 8. Inference Latency

Inference latency measures how long a trained model takes to produce predictions.

| Model | Batch 1 | Batch 128 |
|---|---:|---:|
| MLP | 0.149 ms | 0.703 ms |
| CNN | 0.632 ms | 29.916 ms |
| Attention | 1.458 ms | 19.853 ms |

The MLP was the fastest model at both batch sizes.

Although the CNN had the fewest parameters, it was slower than the MLP. This shows that parameter count alone does not determine inference speed.

At batch size 128, the attention model was faster than the CNN, despite having more parameters.

These results depend on the hardware, software, and measurement conditions used during the experiment.

## 9. Architectural Trade-Offs

**MLP**

The MLP uses dense matrix multiplication. It has a relatively large number of parameters because its neurons are fully connected.

In this experiment, it achieved the lowest inference latency, but its parameter count was much higher than the other models.

**CNN**

The CNN uses shared convolutional filters. This reduces the number of unique weights needed to process an image.

It achieved the highest accuracy and had the fewest parameters. However, convolution operations and intermediate feature maps still require computation and memory access, which can affect latency.

**Attention**

The attention model allows different image patches to exchange information. This helps the model capture relationships between different parts of an image.

However, attention requires additional matrix operations and intermediate attention scores. The computational cost of attention scores increases approximately with the square of the number of tokens.

In this experiment, the attention model did not outperform the CNN or MLP in accuracy.

## 10. Trade-Off Plots

The accuracy-versus-parameters plot shows that the CNN achieved the highest accuracy despite having the fewest parameters.

The MLP had the largest number of parameters but did not achieve the highest accuracy.

The accuracy-versus-latency plot shows that the MLP was the fastest at batch size 1, while the CNN provided the highest accuracy.

These plots demonstrate that there is no single architecture that is best in every category.

The best choice depends on whether the priority is accuracy, model size, inference speed, or computational cost.

## 11. Three Conclusions

1. The CNN gave the best overall balance of accuracy and parameter count. It achieved 89.51% accuracy with only 20,490 parameters.

2. The MLP was the fastest model during inference, even though it had the highest number of parameters. This showed me that having fewer parameters does not necessarily mean a model will run faster.

3. The attention model was useful for understanding how image patches interact, but it achieved the lowest accuracy in this experiment. Its additional operations also showed why attention can be computationally expensive.

## 12. Evidence Files

The following files were generated during the experiments:

- `results/architecture_metrics.csv`
- `results/accuracy_vs_parameters.png`
- `results/accuracy_vs_latency.png`

## 13. Reference

Reddi, V. J., *Introduction to Machine Learning Systems: Principles and Practices of Engineering Artificially Intelligent Systems*, Chapter 4, Sections 4.2, 4.3, 4.5, 4.6, and 4.7.
