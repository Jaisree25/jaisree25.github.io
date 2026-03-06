---
layout: post
title: "The Complete Guide to Neural Networks"
tags: [Neural Network]
---

---

## Table of Contents

1. [The AI Landscape](#the-ai-landscape)
2. [How Machines Learn: Supervised vs Unsupervised Learning](#how-machines-learn-supervised-vs-unsupervised-learning)
3. [From y = mx + b to Neural Networks](#from-linear-regression-to-neural-networks)
4. [What Is a Neural Network?](#what-is-a-neural-network)
5. [A Concrete Example: Predicting Diabetes](#a-concrete-example-predicting-diabetes)
6. [Activation Functions: The Brain's On/Off Switch](#activation-functions-the-brains-onoff-switch)
7. [A Primer on Derivatives](#a-primer-on-derivatives)
8. [Learning From Mistakes: Loss, Backpropagation & Gradient Descent](#learning-from-mistakes-loss-backpropagation--gradient-descent)
9. [Backpropagation: A Full Worked Example](#backpropagation-a-full-worked-example)
10. [Gradient Descent Variants](#gradient-descent-variants)
11. [Building Blocks: Common Layer Types](#common-layer-types)
12. [Skip Connections and ResNets](#skip-connections-and-resnets)
13. [Beyond Binary Classification](#beyond-binary-classification)
14. [Summary Reference Tables](#summary)

---

## The AI Landscape

You've probably heard the terms *artificial intelligence*, *machine learning*, and *deep learning* thrown around — sometimes interchangeably. But what does each one actually mean, and how do they relate to each other?

Artificial intelligence is the broadest term. It refers to any system or technique designed to mimic human intelligence. Under the AI umbrella, two major branches emerge: **rule-based methods** and **machine learning (ML)**.

Rule-based systems operate on explicit if-then logic written by humans — they follow instructions. Machine learning, on the other hand, allows systems to *learn patterns from data*. ML includes techniques like linear regression, support vector machines, and at its deepest level — **deep learning**.

Deep learning uses multi-layered networks to learn increasingly abstract representations of data. And it is within deep learning that **neural networks** live.

![AI/ML Hierarchy Diagram](/assets/images/diagram_ai_hierarchy.svg)

---

## How Machines Learn: Supervised vs Unsupervised Learning

Before diving into neural networks, it's important to understand the two dominant styles of machine learning, because they shape how you design and train a model.

### Supervised Learning

In supervised learning, the training data comes with **labels** — the correct answers. The model is trained to map inputs to those known outputs, and its performance is measured by how close its predictions are to the true labels.

To make this concrete, imagine a spreadsheet of patient health records. Each row contains measurements for one patient — things like blood pressure, cholesterol, BMI, and age. The final column contains a label: `0` meaning the patient does not have diabetes, `1` meaning they do. The model's job is to learn, from thousands of such rows, how to predict that label for a brand new patient it has never seen. The label is the "supervision" — it acts as the teacher, telling the model when it's right or wrong. We will use this exact example throughout the guide to make the mechanics of neural networks as tangible as possible.

Common supervised learning tasks include:
- **Classification** — predicting a category (e.g. spam vs not spam, diabetes vs no diabetes)
- **Regression** — predicting a continuous number (e.g. house price, temperature forecast)

### Unsupervised Learning

In unsupervised learning, there are **no labels**. The model receives only raw input data and must find its own structure, patterns, or groupings within it.

Take the same patient spreadsheet, but remove the diabetes column entirely. Without a label to predict, the model cannot be supervised. However, it might still discover something useful — for instance, that patients naturally cluster into distinct groups based on similar health profiles. This is called **clustering**. Another common technique is **dimensionality reduction**, where a model compresses high-dimensional data into a smaller representation while preserving meaningful structure.

| | Supervised | Unsupervised |
|---|---|---|
| **Labels** | Required | Not present |
| **Goal** | Predict output | Find structure |
| **Examples** | Classification, Regression | Clustering, Dimensionality reduction |
| **Feedback** | Loss against true label | Internal criteria (e.g. cluster tightness) |

Neural networks can be used for both cases, though the most widely understood and deployed applications — including everything covered in this guide — are supervised.

---

## From Linear Regression to Neural Networks

### The Simplest Model: y = mx + b

Before neural networks existed, statisticians used **linear regression** — one of the simplest machine learning techniques. Its formula is:

```
y = mx + b
```

- `x` is the input feature (e.g. a patient's age)
- `m` is the weight (slope) — how much `y` changes per unit of `x`
- `b` is the bias (intercept) — the baseline value of `y` when `x` is zero
- `y` is the prediction

Training this model means finding the values of `m` and `b` that make the line fit the data as well as possible. The result is a straight line drawn through a scatter of data points.

**The key limitation?** Life is not linear. Real-world relationships — like those between health metrics and disease risk — are curved, interacting, and conditional. A single straight line cannot capture them.

### Scaling Up: Multiple Inputs

Now suppose our patient has not one feature but ten. We extend the formula:

```
y = w₁x₁ + w₂x₂ + w₃x₃ + ... + w₁₀x₁₀ + b
```

This is called **multiple linear regression**. It has one weight per feature, plus a bias. It can use all ten health metrics simultaneously to produce a prediction. But it still fits only a flat hyperplane through the data — it cannot model curves or interactions between features.

### The Neural Network as a Massively Extended Version

A neural network takes this idea to its logical extreme. Instead of one set of weights mapping inputs to an output, it stacks *multiple layers* of such computations. Between each layer, it applies a non-linear **activation function** (more on this shortly).

A single hidden layer node does exactly this:

```
z = w₁x₁ + w₂x₂ + ... + wₙxₙ + b    ← same as linear regression
h = activation(z)                  ← this is what makes it powerful
```

With enough nodes and layers, and with non-linear activation functions in between, a neural network can approximate *any* function — no matter how complex.

---

## What Is a Neural Network?

A neural network is a machine learning model inspired by the human brain. Just as the brain is made up of neurons connected by synapses, a neural network has **nodes** (neurons) connected by **weights** (synaptic strengths).

These nodes are organized into three types of layers:

- **Input layer** — receives the raw data, one node per feature
- **Hidden layers** — where the network learns intermediate representations
- **Output layer** — produces the final prediction

There are many types of neural networks, each suited to different tasks:

| Type | Best For |
|---|---|
| **MLP** (Multilayer Perceptron) | Structured / tabular data |
| **CNN** (Convolutional Neural Network) | Image processing & classification |
| **RNN** (Recurrent Neural Network) | Sequential data (e.g. time series, audio) |
| **Transformer** | Language, sequences, generative AI |

>**Note**: An MLP is the foundational type of neural network: fully-connected layers of nodes stacked in sequence, with no special structure. Every other architecture (CNNs, RNNs, Transformers) is a neural network that adds constraints or structure on top of this basic idea. When people say "neural network" without qualification, they usually mean an MLP.

---

## A Concrete Example: Predicting Diabetes

Now that we understand what a neural network is structurally, let's bring it to life with the patient health example introduced earlier. Recall the spreadsheet: each row is a patient, with **10 health metric columns** — blood pressure, cholesterol, BMI, age, and so on — and a final label column (`0` = no diabetes, `1` = diabetes). We want to train an MLP that, given a new patient's 10 measurements, predicts whether they have diabetes.

### Setting Up the Network

Since there are 10 features, the **input layer has 10 nodes** — one per feature. The output layer has **1 node** outputting a probability between 0 and 1.

### The Forward Pass

When we feed a patient's row into the network, the data flows forward through each layer in sequence. At every hidden layer node, the incoming values are:

1. Multiplied by their respective weights
2. Summed together
3. Added a bias term
4. Passed through an **activation function**

At the output layer, a **sigmoid** activation squashes the result to a probability. An output of `0.85` means an estimated **85% chance** the patient has diabetes.

This end-to-end journey from input to prediction is called the **forward pass**.

![Neural Network Architecture Diagram](/assets/images/diagram_neural_network.svg)

---

## Activation Functions: The Brain's On/Off Switch

Activation functions are arguably the most important ingredient in a neural network. Without them, no matter how many layers you stack, the entire network collapses into a single linear transformation — functionally identical to plain linear regression. Let's understand exactly why they matter.

### The Biological Analogy

In the brain, a biological neuron only fires — passes a signal to the next neuron — when its incoming signals are strong enough to exceed a threshold. Below the threshold, it stays silent. Activation functions mimic this behaviour: they decide **how strongly a node should "fire"**, i.e. how large the output signal passed to the next layer should be.

Think of each activation function as a **gate**. It receives a weighted sum of inputs and then determines:
- Whether the neuron activates at all
- If so, with what strength

### Why Non-Linearity Is Essential

Consider what happens without any activation function. Each layer performs:

```
output = W · input + b
```

This is a linear operation. If you stack two such layers:

```
Layer 2(Layer 1(x)) = W₂(W₁x + b₁) + b₂ = (W₂W₁)x + (W₂b₁ + b₂)
```

The result is still just `Wx + b` — a single linear layer. No matter how deep the network goes, it can only ever model a straight line through data. You lose all expressive power of the depth.

When you insert a non-linear activation function between layers, the combined function becomes genuinely more complex with each added layer. This is what allows neural networks to model curves, decision boundaries, and intricate real-world patterns.

### Activation Functions in Practice: Where to Use What

A useful mental model: **hidden layers** need activation functions that allow rich learning; **output layers** need activation functions whose range matches the prediction task.

- **After every hidden layer** → use **ReLU** by default
- **Output for binary classification** → use **Sigmoid** (outputs a probability between 0 and 1)
- **Output for multi-class classification** → use **Softmax** (outputs a probability distribution over all classes)
- **Output for regression** → use **no activation** (linear), since the output can be any real number
- **Skip connections in ResNets** → use **no activation** (linear identity pass-through)

There is also a notable exception: **ResNet-style skip connections** intentionally bypass the activation function, passing the raw input directly to a later layer. We'll cover this in detail in the ResNet section.

![Activation Functions Diagram](/assets/images/diagram_activation_functions.svg)

### ReLU (Rectified Linear Unit)

```
f(x) = max(0, x)
```

ReLU is the default choice for hidden layers. It is computationally simple: it outputs `x` if `x` is positive, and `0` otherwise. This zero-ing out of negative values is what introduces non-linearity.

**Why ReLU over functions like Tanh?** It solves the **vanishing gradient problem**. In deep networks, gradients are multiplied together as they propagate backwards. Sigmoid and Tanh squash their outputs into small ranges ([0,1] and [−1,1] respectively), so their gradients are always less than 1. Multiplying many small gradients together causes the product to shrink exponentially as it travels toward earlier layers — the gradient effectively *vanishes*, and early layers stop learning. ReLU has a gradient of exactly `1` for all positive inputs, so it does not diminish the gradient as it passes through, keeping early layers trainable in deep networks.

A minor drawback: if a node's input is consistently negative, ReLU outputs 0 and the gradient is also 0 — the node never activates or learns. This is called the **dying ReLU problem**. **Leaky ReLU** addresses this by using a small slope (typically 0.01) for negative inputs instead of flat zero.

### Sigmoid

```
f(x) = 1 / (1 + e⁻ˣ)
```

Sigmoid squashes any input to a value between 0 and 1 — making it a natural probability. It is the standard choice for the **output node of a binary classification problem**. Its smooth S-curve is differentiable everywhere.

The downside in hidden layers is the vanishing gradient: for large positive or negative inputs, the sigmoid saturates (output approaches 0 or 1), and its gradient approaches 0. This makes it a poor choice for hidden layers in deep networks.

### Tanh (Hyperbolic Tangent)

```
f(x) = (eˣ − e⁻ˣ) / (eˣ + e⁻ˣ)
```

Tanh is similar to sigmoid but maps inputs to the range (−1, 1). Because tanh outputs both positive and negative values, its gradients can push weights in either direction during training — unlike sigmoid, which only outputs positive values and causes weight updates to all move the same way, making it preferable to sigmoid in hidden layers. However, it still saturates and causes vanishing gradients in very deep networks, so ReLU is usually preferred.

### Softmax

```
f(xᵢ) = eˣⁱ / Σⱼ eˣʲ
```

Softmax is used at the **output layer of multi-class classification** problems. It takes a vector of raw scores (called logits) and converts them into a probability distribution: each output is between 0 and 1, and all outputs sum to exactly 1.0. The class with the highest probability is the model's prediction.

For example, in an image classifier with 5 animal classes, softmax might output `[cat: 0.03, dog: 0.12, bird: 0.78, fish: 0.04, frog: 0.03]`.

### The Differentiability Requirement

There is a mathematical requirement that is easy to overlook: activation functions must be **differentiable** (or at least differentiable almost everywhere). The reason is backpropagation. When the network computes gradients to update weights, it uses the **chain rule of calculus**, which requires multiplying derivatives at each layer. If an activation function has no derivative, the gradient cannot flow backwards through it, and learning breaks down.

This is why a simple step function (0 if input < 0, else 1) is not used in practice — it is not differentiable at 0 and has zero gradient everywhere else, meaning the network would receive no learning signal. ReLU is technically not differentiable exactly at zero, but it has well-defined derivatives everywhere else and works extremely well in practice.

---

## A Primer on Derivatives

![Limit Derivative Diagram](/assets/images/diagram_limit_derivative.svg)

Before we can fully understand backpropagation, we need to understand **derivatives**. If you're already comfortable with calculus, feel free to skim this section — but if not, this foundation will make the rest of the guide click.

### What Is a Derivative?

A derivative measures **how much a function's output changes when you change its input by an infinitely small amount**. Informally, it's the *slope* of a function at a specific point.

For a simple function like:

```
f(x) = x²
```

The derivative is written as `f'(x)` or `df/dx`, and it equals `2x`. This tells us:

- At `x = 3`, the slope is `2(3) = 6` — for a tiny increase in `x`, the output increases roughly 6 times as fast
- At `x = 0`, the slope is `0` — the function is at a flat minimum there
- At `x = −2`, the slope is `−4` — the function is decreasing as `x` increases

### Derivative Rules You'll Need

**Power rule:**
```
d/dx [xⁿ] = n · xⁿ⁻¹

Examples:
  d/dx [x²]  = 2x
  d/dx [x³]  = 3x²
  d/dx [5x]  = 5
  d/dx [c]   = 0   (c is a constant)
```

**Chain rule** — used when functions are nested inside each other:
```
If y = f(g(x)),  then  dy/dx = f'(g(x)) · g'(x)

Example:
  y = (3x + 1)²
  Let u = 3x + 1, so y = u²
  dy/du = 2u,   du/dx = 3
  dy/dx = 2u · 3 = 6(3x + 1)
```

The chain rule is the mathematical foundation of backpropagation. Because a neural network is a deeply nested composition of functions, computing the gradient of the loss with respect to an early-layer weight requires multiplying the chain of local derivatives all the way from the output back to that weight.

### Derivatives of Common Activation Functions

These are the derivatives you'll encounter in backpropagation:

**ReLU:**
```
f(x)  = max(0, x)
f'(x) = 1  if x > 0
        0  if x ≤ 0
```
The derivative is simply 1 for positive inputs and 0 otherwise — a perfect pass-through for positive gradients.

**Sigmoid:**
```
f(x)  = σ(x) = 1 / (1 + e⁻ˣ)
f'(x) = σ(x) · (1 − σ(x))
```
If σ(x) = 0.8, then f'(x) = 0.8 × 0.2 = 0.16. Note that this is always between 0 and 0.25, which is why gradients shrink when passing through many sigmoid layers.

**MSE Loss (Looking Ahead to Loss Functions):**
```
L = (ŷ − y)²
dL/dŷ = 2(ŷ − y)
```

---

## Learning From Mistakes: Loss, Backpropagation & Gradient Descent

### The Loss Function

To measure how wrong a prediction is, we use a **loss function**. It calculates the error between the prediction `ŷ` and the true label `y`.

**Mean Squared Error (MSE)** — used for regression:
```
L = (ŷ − y)²
```

**Binary Cross-Entropy** — used for binary classification (e.g. our diabetes example):
```
L = −[y·log(ŷ) + (1−y)·log(1−ŷ)]
```

The bigger the loss, the further off the model is. Training aims to minimize this value.

### Backward Pass (Backpropagation)

Once the loss is computed, the network needs to figure out which weights contributed to the error and by how much. This is done through **backpropagation** — the process of applying the chain rule recursively from the output layer back to the input layer.

At each layer, backpropagation computes the **gradient** of the loss with respect to every weight. A gradient is the generalization of a derivative to multi-variable functions — it points in the direction of steepest increase in loss.

### Gradient Descent

![Gradient Descent Diagram](/assets/images/gradient_descent.png)
*Image from analyticsvidhya*

With gradients computed, each weight is updated using the **gradient descent** rule:

```
W_new = W_old − η · (∂L / ∂W_old)
```

- **W** — the weight being updated
- **η (eta)** — the *learning rate*, controlling step size
- **∂L/∂W** — the gradient of the loss with respect to the weight

The sign of the gradient tells the network which direction to move:

| Gradient sign | Meaning | Action |
|---|---|---|
| Positive | Increasing W increases loss | Decrease W |
| Negative | Increasing W decreases loss | Increase W |
| Near zero | W is near-optimal | Small / no change |

---

## Backpropagation: A Full Worked Example

Let's walk through a minimal network step by step with real numbers, so the mathematics becomes fully concrete.

**The network:**
- 1 input node: `x = 2`
- 1 hidden node with **ReLU** activation
- 1 output node (no activation — regression output)
- Weights: `w₁ = 0.5` (input → hidden), `w₂ = 0.8` (hidden → output)
- No bias terms for simplicity
- Loss: **MSE** → `L = (ŷ − y)²`
- True label: `y = 1.0`
- Learning rate: `η = 0.1`

![Backpropagation Worked Example](/assets/images/diagram_backprop_math.svg)

### Weight Updates

With `η = 0.1`:

```
w₂_new = w₂ − η · dL/dw₂ = 0.8 − 0.1 × (−0.4) = 0.84
w₁_new = w₁ − η · dL/dw₁ = 0.5 − 0.1 × (−0.64) = 0.564
```

Both weights increased — which makes sense, because both gradients were negative, meaning increasing the weights would reduce the loss. If we ran the forward pass again with these new weights:

```
z₁ = 2 × 0.564 = 1.128
h  = ReLU(1.128) = 1.128
ŷ  = 1.128 × 0.84 = 0.948
L  = (0.948 − 1.0)² = 0.0027
```

The loss dropped from `0.04` to `0.0027` after just one update. Over many iterations, the network converges toward `ŷ ≈ 1.0`.

---

## Gradient Descent Variants

In the example above, we updated weights after computing the gradient on a single data point. In practice, there are three strategies for choosing *how much data to use* before each weight update. In other words **when to update weights**.


### Batch Gradient Descent

Compute the gradient using the **entire dataset** before making a single weight update. This produces a very accurate gradient estimate, leading to smooth, stable convergence. However, it requires loading all the data into memory at once and is extremely slow for large datasets with millions of rows.

### Stochastic Gradient Descent (SGD)

Compute the gradient and update weights using **one randomly selected sample** at a time. This is extremely fast and updates happen frequently, but the gradient estimate is very noisy — the loss curve zigzags unpredictably rather than descending smoothly. The noise can sometimes be beneficial, helping the model escape shallow local minima.

### Mini-Batch Gradient Descent

A compromise: compute the gradient over a **small random batch** of samples (typically 32, 64, 128, or 256 rows) and update weights after each batch. This is the standard approach used in virtually all modern deep learning. It:

- Is fast and memory-efficient
- Produces a reasonably accurate gradient estimate
- Works naturally with GPU parallelism, which can process an entire batch simultaneously

---

## Common Layer Types

Neural networks are assembled from specialized building blocks. Each layer type is designed to exploit a particular structure in the data or to regularize the model during training.

![Common Layer Types](/assets/images/diagram_layer_types.svg)

### Dense (Fully Connected) Layer

The fundamental building block we've used throughout this guide. Every node in the layer is connected to every node in the next layer. Each connection has its own learnable weight. Dense layers are used when the input features don't have a meaningful spatial or sequential structure — such as tabular data (like our diabetes CSV).

### Convolutional Layer (Conv2D)

A convolutional layer applies a small learnable **filter** (e.g. 3×3 pixels) that slides across the input — typically an image — and computes a dot product at each position. This detects local spatial patterns: edges, corners, textures at early layers, and progressively more abstract features (eyes, wheels, fur) at deeper layers.

A key advantage of convolutions is **parameter sharing**: the same filter is reused across all positions of the image, dramatically reducing the number of weights compared to a fully connected layer applied to an image. A 256×256 image as input to a dense layer would require 65,536 weights per node; a single 3×3 convolutional filter requires only 9.

### Pooling Layer (MaxPool / AvgPool)

Pooling layers reduce the spatial dimensions of the feature maps produced by convolutional layers. **Max pooling** takes the maximum value within each small patch (e.g. 2×2), discarding fine spatial detail while retaining the strongest activations. **Average pooling** takes the average instead.

Pooling serves two purposes: it reduces computational load and memory usage, and it makes the network partially **translation-invariant** — a slightly shifted object in an image still produces similar feature activations after pooling.

### Flatten Layer

A flatten layer reshapes a multi-dimensional tensor into a 1D vector. It is used as the **bridge between convolutional layers and dense layers** in image classification networks. After a CNN has extracted spatial features from an image through convolutional and pooling layers, the flatten layer unrolls those features into a vector that a dense layer can process.

### Dropout Layer

Dropout is a **regularization technique** that randomly sets a fraction of node outputs to zero during training (typically 20–50%). Each training step, a different random subset of nodes is silenced. At inference time, dropout is disabled and all nodes are active.

Dropout prevents overfitting by forcing the network to learn redundant representations — no single node can be relied upon to carry all the information. It acts like training an ensemble of many slightly different sub-networks simultaneously.

### Batch Normalization

Batch normalization normalises the activations of a layer across the current mini-batch to have approximately zero mean and unit variance. It is typically inserted **before or after the activation function** in hidden layers.

Benefit of batch normalization:
- **Stabilizes training** by keeping the inputs to each layer on a consistent scale as weights update

---

## Skip Connections and ResNets

As neural networks get deeper — tens or even hundreds of layers — a new problem emerges: gradients struggle to propagate all the way back to early layers during backpropagation. Even with ReLU, the repeated multiplication of gradients across many layers can cause them to shrink toward zero, starving early layers of the learning signal. This is called the **vanishing gradient problem** in deep networks.

**Residual Networks (ResNets)**, introduced by He et al. in 2015, solved this with a simple but powerful idea: **skip connections** (also called residual connections or identity shortcuts).

Instead of a block of layers learning the function `F(x)`, the block learns the *residual* `F(x)`, and the input `x` is **added directly** to the block's output:

```
output = F(x) + x
```

This addition bypasses the block entirely. The gradient during backpropagation can now flow through two paths: through the layers in the block, *or* directly through the skip connection. The skip connection has a gradient of exactly `1` with respect to `x` (since `d/dx [F(x) + x] = F'(x) + 1`), which ensures that even if `F'(x)` vanishes, a gradient of `1` always flows back unimpeded.

Skip connections also make it easier for layers to learn identity mappings (i.e. "don't change anything here"), which is useful when a layer doesn't need to transform the data further.

The skip connection passes the input through untouched — no activation, no transformation. The conv layers and the skip connection are then added together, and a single activation is applied to the combined result.

ResNets enabled training of networks with 50, 100, even 152 layers — depths that were previously impossible to train reliably. They are foundational to modern computer vision and influenced architecture design across all of deep learning.

---

## Beyond Binary Classification

So far we've focused on a binary classification problem (diabetes: yes or no). Neural networks are far more flexible. The same principles apply to:

- **Multi-class classification** — the output layer uses Softmax over N nodes, one per class (e.g. recognising 1,000 ImageNet categories)
- **Regression** — the output layer has one (or more) nodes with no activation function, predicting continuous values
- **Image processing** — CNNs apply convolutional + pooling layers to extract spatial hierarchies of features before a Dense output
- **Sequence modelling** — RNNs and Transformers process variable-length sequences

One key limitation of vanilla MLPs is that they produce a single output for a single input. This is what motivated **Recurrent Neural Networks (RNNs)** and later **Transformers** — architectures capable of processing and producing *sequences* of data, which brought upon modern NLP, speech recognition, and generative AI.

---

## Summary

### Architecture Decisions

| Task | Output activation | Loss function |
|---|---|---|
| Binary classification | Sigmoid | Binary cross-entropy |
| Multi-class classification | Softmax | Categorical cross-entropy |
| Regression | None (linear) | MSE |

### Activation Function Quick Reference

| Function | Range | Typical use | Key property |
|---|---|---|---|
| ReLU | [0, ∞) | Hidden layers | Solves vanishing gradient |
| Leaky ReLU | (−∞, ∞) | Hidden layers | Fixes dying ReLU |
| Sigmoid | (0, 1) | Binary output | Output is probability |
| Tanh | (−1, 1) | Hidden layers | Zero-centred |
| Softmax | (0, 1), sums to 1 | Multi-class output | Probability distribution |
| Linear | (−∞, ∞) | Regression output, skip connections | No transformation |

### Core Concepts

![Forward & Backward Pass Diagram](/assets/images/diagram_forward_backward.svg)

| Concept | What It Does |
|---|---|
| **Forward Pass** | Data flows input → output to produce a prediction |
| **Loss Function** | Measures how wrong the prediction is |
| **Backpropagation** | Chain rule applied backwards; computes gradients for every weight |
| **Gradient Descent** | Updates weights in the direction that reduces loss |
| **Activation Function** | Introduces non-linearity; acts as a gate on each neuron's output |
| **Dropout** | Regularization by randomly silencing nodes during training |
| **Batch Normalization** | Normalises activations for stable, faster training |
| **Skip Connection** | Bypasses layers to preserve gradient flow in deep networks |