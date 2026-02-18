La belleza del cross entropy es que además de penalizar malas predicciones lo hace más duro si son overconfident


### Compute and memory estimation

To estimate the amount of compute and memory needed for a performant training of this architecture let's detail the calculations starting with the estimated FLOPs required to train the model. According to [Training compute-optimal Large Language Models](https://arxiv.org/pdf/2203.15556). 

For training, we'll use FP32 precision.

$FLOPs ≈ 6 × N × D$

Where: 
- N = Number of parameters in the model
- D = Number of tokens in training dataset

$FLOPs ≈ 6 × 9M × 12M = 648B FLOPs = 648 × 10^12 FLOPs = 648 TFLOPs$

Using an [Nvidia GeForce RTX 4090](https://www.nvidia.com/en-us/geforce/graphics-cards/40-series/rtx-4090/) with approximately 83 TFLOPS in CUDA Cores and a similar capacity on AI Tensor Cores (which will be the primary compute, given the requirements of attention mechanisms in Alexandria). Assuming an utilization of 40% of the FLOPs we can perform the next calculation:

648 TFLOPs / 33 TFLOPs = 21.38 seconds

However

In RunPod RTX 4090 costs 0.59 USD per hour. Hence total training might cost aorund 7.08 USD. This costs considers a single epoch.

$Memory ≈ 4 × params = 4 x 9M = 36MB$
