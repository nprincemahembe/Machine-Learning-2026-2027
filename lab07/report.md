# Lab 7: Compression

## Objective

This laboratory investigates pruning and knowledge distillation on Fashion-MNIST.

## Experimental setup

- Device: cpu
- Seed: 42
- Training samples: 60,000
- Test samples: 10,000
- Batch size: 128
- Optimizer: Adam
- Learning rate: 0.001

## Baseline CNN

- Parameters: 824,458
- Test accuracy: 92.04%
- Latency: 1.4955 ms
- Dense size: 3,300,416 bytes

## Experimental results

| Model | Accuracy | Parameters | Latency (ms) | Dense MiB | Sparse MiB | MACs | Energy (kWh) |
|---|---:|---:|---:|---:|---:|---:|---:|
| teacher | 92.04% | 824,458 | 1.4955 | 3.148 | 16.009 | 4,643,840 | 0.002641 |
| global_0 | 91.72% | 824,458 | 1.5054 | 3.148 | 16.009 | 4,643,840 | 0.001517 |
| global_25 | 91.91% | 824,458 | 1.5297 | 3.148 | 12.055 | 4,643,840 | 0.001528 |
| global_50 | 92.26% | 824,458 | 1.4859 | 3.148 | 8.098 | 4,643,840 | 0.001534 |
| global_75 | 92.36% | 824,458 | 1.5551 | 3.148 | 4.117 | 4,643,840 | 0.001394 |
| global_90 | 92.52% | 824,458 | 1.6014 | 3.148 | 1.691 | 4,643,840 | 0.001335 |
| global_95 | 92.22% | 824,458 | 1.4872 | 3.148 | 0.866 | 4,643,840 | 0.001354 |
| layerwise_0 | 91.72% | 824,458 | 1.4709 | 3.148 | 16.009 | 4,643,840 | 0.001385 |
| layerwise_25 | 92.16% | 824,458 | 1.3805 | 3.148 | 12.008 | 4,643,840 | 0.001432 |
| layerwise_50 | 92.19% | 824,458 | 1.4795 | 3.148 | 8.007 | 4,643,840 | 0.001326 |
| layerwise_75 | 92.21% | 824,458 | 1.5154 | 3.148 | 4.006 | 4,643,840 | 0.001323 |
| layerwise_90 | 89.17% | 824,458 | 1.4635 | 3.148 | 1.605 | 4,643,840 | 0.001387 |
| layerwise_95 | 86.94% | 824,458 | 1.6148 | 3.148 | 0.805 | 4,643,840 | 0.001577 |
| structured_10 | 92.08% | 745,864 | 1.6678 | 2.848 | 14.459 | 3,901,784 | 0.001423 |
| structured_25 | 91.87% | 615,594 | 1.1897 | 2.351 | 11.901 | 2,806,144 | 0.001106 |
| structured_50 | 91.98% | 409,034 | 0.9390 | 1.563 | 7.873 | 1,420,032 | 0.000746 |
| plain_student | 89.83% | 105,866 | 0.7758 | 0.406 | 2.093 | 1,117,056 | 0.001111 |
| distilled_student | 89.95% | 105,866 | 0.5443 | 0.406 | 2.093 | 1,117,056 | 0.002110 |

## Pruning and storage

Global pruning ranks weights across all layers. Layer-wise pruning ranks weights independently within each layer.

Fixed masks are maintained during three epochs of fine-tuning.

Unstructured pruning creates zero weights without reducing the dimensions of dense tensors. Therefore, ordinary dense CPU kernels may still perform the same operations.

Sparse COO storage includes index overhead and may be larger than dense storage at low sparsity.

Structured pruning physically removes filters and adjusts the connected layers. This reduces model dimensions and nominal compute.

Structured removal targets refer to filters, not the percentage of remaining zero weights.

## Knowledge distillation

Temperature = 4; alpha = 0.5.

The plain and distilled students use the same architecture, initialization seed and five-epoch training budget.

The implementation uses PyTorch KL divergence with teacher probabilities as the target, corresponding to KL(teacher || student).

Plain student accuracy: 89.83%.

Distilled student accuracy: 89.95%.

Distillation accuracy difference: +0.12 percentage points.

## Efficiency pillars

- Data: identical Fashion-MNIST datasets.
- Model: parameter counts and serialized size.
- Compute: nominal MACs and approximate FLOPs.
- Energy: estimated training energy from CodeCarbon.

Energy values describe the individual training or fine-tuning runs, not total lifecycle energy. Teacher training energy is excluded from student and pruning runs.

CodeCarbon energy values are estimates, not direct power-meter measurements.

Latency is measured with batch size 1 after 30 warm-up iterations, using the median of 100 timed iterations.

Approximate FLOPs use two operations per MAC. Dense compute estimates do not assume sparse hardware acceleration.


## Conclusion

The results showed that model compression can reduce storage requirements and computational costs while maintaining good accuracy.

Global pruning performed better than layer-wise pruning at high sparsity levels. The highest accuracy was achieved with 90% global pruning, which reached 92.52%, compared to the baseline accuracy of 92.04%. Even at 95% global sparsity, the model maintained an accuracy of 92.22%. However, additional fine-tuning may also have contributed to these improvements.

Unstructured pruning did not provide a consistent improvement in inference latency because the dimensions of the weight tensors remained unchanged. However, sparse storage was useful at high sparsity levels. At 95% global sparsity, the serialized storage size was reduced by approximately 72.48% compared to the dense representation.

Structured pruning provided a better balance between accuracy and computational efficiency. Removing 50% of the filters reduced the parameter count by 50.39%, computational requirements by 69.42%, and measured inference latency by approximately 37.21%. The accuracy decreased by only 0.06 percentage points compared to the baseline.

Knowledge distillation improved student accuracy from 89.83% to 89.95% without increasing the number of parameters or model size. However, the distilled student required approximately 90% more estimated training energy than the plain student. Since both students had the same architecture, their measured latency difference may be due to hardware and benchmarking variability rather than distillation itself.

Overall, 50% structured pruning provided the best balance between accuracy, model size and computational efficiency in this experiment. The student models were more suitable when minimizing model size was the main priority.

These experiments demonstrate that reducing the number of nonzero weights does not necessarily improve hardware performance. The choice of compression technique should consider accuracy, storage, latency, compute and energy together.

## Limitations

The experiments were conducted on a CPU using a single random seed. Energy consumption was estimated using CodeCarbon rather than measured directly with a power meter. Therefore, small differences in accuracy and latency should be interpreted cautiously.
## References

Vijay Janapa Reddi, Introduction to Machine Learning Systems: Principles and Practices of Engineering Artificially Intelligent Systems.

Chapter 9: Efficient AI.

Chapter 10: Sections 10.3 and 10.4.
