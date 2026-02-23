# Training pipeline design desicions

In this document the different design desicions to train Alexandria are discussed, detailed and documented.

## Hardware

Even though Alexandria code has been developed entirely in my local machine, I decided to train the model using a GPU on a remote server. At first, to practice the process in a different service. 

For this training, I decided to run the process in a Pod in RunPod. The chosen GPU was a Nvidia GeForce RTX 4090.

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

$Memory ≈ 4 × params = 4 x 9M = 36MB$

However, this estimation is a good baseline to start, but final process took around 3 minutes. 

As the use of this GPU in RunPod costs 0.59 USD per hour, it is expected to use the compute power of this GPU and spend only 3 cents.

### Why did the final training took longer than expected?

In general, some implementation decisions drove to this result. The majority of the compute time was used by the evaluation steps.
The evaluation step happened at every 100 steps during training, with a total of 25 times. 

Evaluation was performed over checkpoints in eval mode. However, no alternative GPU or compute node has been designed for this task. Meaning that training stopped completely during evaluation and then continued from last time. Adding the total time each evaluation took on the model we get the final 3 minutes for the whole flow.

Anyway, this does not represent a problem in my case, but it states the need for a separate compute for bigger training processes to run evaluation separately and use all the possible hardware power only for training.

## Metrics and artifacts logging

To track performance metrics on the model and logging checkpoints and the final version of the weights I decided to use Comet, primarily for its simple integration with Pytorch and because I wanted to keep the logs in a separate service than the Pod used.

The information tracked and sent to Comet included:

- Cross-entropy loss in train and test datasets
- Perplexity in train and test datasets
- Examples of generated text using greedy sampling by different checkpoints
- Checkpoints of the weights each 100 steps
- Carbon emissions generated during training and calculated with codecarbon
- The final weights of the model

## RunPod pod configuration

Pod image: runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404

GPU: [Nvidia GeForce RTX 4090](https://www.nvidia.com/en-us/geforce/graphics-cards/40-series/rtx-4090/)

# The training process

## Batch size

Alexandria was trained on 2500 batches of 32 examples per batch. In general, the whole data could have been fitted into the VRAM available in the RTX 4090. However, I decided to process all in batches as it would lead to a more stable learning process for the model. 

## Optimizer and scheduler

Adam with Weight Decay has been selected as the optimizer algorithm for training.

For scheduling process a cosine anealling LR component was selected.

## Loss function

The implementation was based on following cross-entropy. The selection for this metrics comes given two factors: 

1. Text generation process acts as a classification problem, in general, we're trying to select the most probable token out of a fixed vocabulary, or, in other words, we're predicting a class that comes based on the past. Cross-entropy is a great metric for classification problems, as it punishes overconfident errors harder.
2. When working with generative models, what we're trying to do is actually modeling a probability distribution that is similar enough to the true probability distribution of real data. In this case, I want Alexandria to model a probability distribution of Spanish language that resemble the one present in its training data. For this, some metrics can help us comparing the similarity between distributions, such as KL-divergence. It turns out, cross-entropy actually optimizes KL-divergence as well, making it a great metric for us to use as loss function.

Cross-entropy formula

$$\mathcal{L} = -\frac{1}{N}\sum_{i=1}^{N}\sum_{c=1}^{C} p_{ic} \log({q}_{ic})$$

Cross-entropy defined with KL-divergence

$$\mathcal{L} = D_{KL}(p \| q) + H(p)$$

$$\text{where:}$$

$$H(p) = -\sum_{x} p(x) \log p(x)$$

$$D_{KL}(p \| q) = \sum_{x} p(x) \log \frac{p(x)}{q(x)}$$

# Evaluation process

Each 100 steps an evaluation process runs in the same GPU used for training. This does not represent a big issue on performance given the compute requirements of our model, but it would be recommended to use separate compute nodes to utilize the max capacity of the GPU only for training.

Each evaluation consists on a calculation of the loss and perplexity on a test set, consisting on 675 batches or around 21,600 examples. Around 20% of the total tiny-coop-es dataset.

After this, an sample of text was generated using the next setup:
- Prompt: "Un conejo y una tortuga se encontraban "
- Max tokens : 30
- Temperature: 1.0

and logged to Comet, altogether with a checkpoint that generated that text and the evaluation metrics.