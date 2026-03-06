---
layout: post
title: "Action Chunking Transformer"
tags: [Robotics, Transformers, Imitation Learning]
---

---

## Table of Contents

1. [What Is ACT?](#what-is-act)
2. [A Quick Refresher: Transformers](#a-quick-refresher-transformers)
3. [The Task and the Data](#the-task-and-the-data)
4. [What ACT Actually Does](#what-act-actually-does)
5. [ACT Architecture Overview](#act-architecture-overview)
6. [The VAE Encoder](#the-vae-encoder)
7. [The Main Encoder](#the-main-encoder)
8. [The Main Decoder](#the-main-decoder)
9. [Inference: What Happens at Test Time](#inference-what-happens-at-test-time)
10. [Summary](#summary)

---

## What Is ACT?

In robotics, a policy is simply the function that maps what the robot currently observes to what action it should take next. This post covers **ACT — Action Chunking Transformer**, a policy architecture for robot learning introduced in the paper *"Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware"* by Tony Z. Zhao, Vikash Kumar, Sergey Levine, and Chelsea Finn. 

ACT is an example of **imitation learning**: rather than programming a robot with explicit rules, we show it how to perform a task by recording human demonstrations, and train a neural network to replicate that behaviour. The network learns to map what the robot currently sees and feels — camera images and joint states — to a sequence of future actions.

Throughout this post, we will ground the explanation in a single concrete task: **a robotic arm picking up an object and dropping it into a box**.

---

## A Quick Refresher: Transformers

ACT is built on the transformer architecture, so a brief recap is useful before diving in.

A transformer is a type of neural network designed to process and produce **sequences of information**. It is made up of three core components repeated across multiple layers:

- **Embedding layer** — converts raw inputs (tokens, sensor readings, image patches) into dense vector representations
- **Attention layer** — allows each element in the sequence to look at every other element and decide which ones are most relevant to it
- **MLP layer** — a standard feedforward network applied to each position independently

The defining feature of transformers is the **attention mechanism**. In short, attention allows the model to learn *context* — a word's meaning depends on the words around it, just as a robot's next action depends on its current state and recent history. This makes transformers exceptionally powerful for any problem where relationships between elements in a sequence matter.

Transformers were originally developed for language tasks — translation, sentiment analysis, next-word prediction. ACT applies the same machinery to robotics, where the "sequence" is a series of robot actions over time.

---

## The Task and the Data

Before training any neural network, we need data. For our task — picking up an object and dropping it into a box — data collection works as follows.

Several cameras are positioned around the robot arm to capture video from multiple viewpoints. A human operator then **teleoperates** the arm, physically controlling it to perform the task repeatedly while the cameras record. In addition to the video, the joint and motor states of the arm are logged at each timestep — these capture the precise configuration and movement of the robot.

![Data Pipeline Diagram](/assets/images/diagram_data_pipeline.svg)

The result is a dataset of many demonstration episodes. Each episode is a sequence of (image frames, joint states, actions) tuples. This dataset is what ACT is trained on.

> **Note:** Human demonstrations are imperfect. Operators may move slightly inconsistently across different recordings of the same task — a small shake here, a slightly different grip angle there. A robust policy must learn the underlying task, not overfit to these human inconsistencies.

---

## What ACT Actually Does

It helps to be precise about ACT's inputs and outputs before examining the architecture.

At each timestep, ACT receives:
- **Camera images** — the current visual state of the scene
- **Current robot joint states** — the robot's physical configuration right now
- **Past actions** — what the robot has done recently

From these, ACT predicts: **a short sequence of future joint actions** — not just the next single action, but a chunk of several actions ahead.

This is the "action chunking" in the name, and it is a deliberate design choice. Real-world robot movements are smooth and continuous — the arm does not jerk from position to position one step at a time. By predicting a whole chunk of actions at once, the policy enforces **temporal coherence**: the predicted sequence is internally consistent rather than composed of individually-noisy single-step predictions. Chunking also reduces the number of forward passes through the model needed to complete a task, which saves computation.

![Action Chunking Diagram](/assets/images/diagram_action_chunking.svg)

---

## ACT Architecture Overview

ACT has three main components:

| Component | Role | Active During |
|---|---|---|
| **VAE Encoder** | Introduces controlled variation to prevent overfitting | Training only |
| **Main Encoder** | Produces a rich contextual representation of the current state | Training & inference |
| **Main Decoder** | Predicts the sequence of future actions | Training & inference |

The VAE encoder is only used during training. At inference time, it is replaced by a fixed zero vector. We will explain exactly why below.

---

## The VAE Encoder

The VAE (Variational Autoencoder) Encoder is the most conceptually novel component of ACT. Its purpose is to prevent the model from **overfitting to the imperfections in human demonstrations**.

### The Problem It Solves

Human teleoperators are not perfectly consistent. Across many recordings of the same task, the same action might be executed with slightly different timing, speed, or path. If we train the model to exactly reproduce these demonstrations, it will learn to mimic the noise as well as the task. The VAE encoder addresses this by injecting learned, controlled randomness into the training process.

### The CLS Token

The input to the VAE encoder is:

```
[CLS, joint states, image features, past actions]
```

`image features` here refers to the output of a **CNN** (Convolutional Neural Network) applied to the camera frames — the raw images are first compressed into compact feature vectors before being passed to the transformer.

`CLS` is the **classification token** — a single learned vector that the encoder uses to summarise the entire input sequence. It begins as a randomly initialized vector and is updated through training. The CLS token's final representation is used as the basis for generating the latent distribution.

> **Note:** Past actions are represented as a matrix of numbers — one row per recent timestep, one column per joint. For a robot with 6 joints tracking the last 5 timesteps, this is a 5×6 matrix where each value is a joint angle. This matrix is embedded into tokens before being passed into the VAE encoder alongside the CLS token, joint states, and image features.

### The Latent Distribution

Rather than producing a single fixed summary of the input, the VAE encoder produces a **probability distribution** — a range of possible summaries, each slightly different. This is what introduces controlled variation.

Specifically, the encoder uses the CLS token's output to predict two vectors via learned weights:

- **μ (mu)** — the mean of the distribution
- **log(σ²)** — the log variance of the distribution

Together, μ and σ define a Gaussian distribution in latent space. During training, we **sample** a latent vector `z` from this distribution rather than using a single fixed value. Think of `z` as a compressed, slightly-varied summary of the current situation and past actions.

By sampling from a distribution rather than computing a deterministic encoding, the model is forced to be robust to small variations in input — it cannot over-rely on exact reproduction of any one demonstration.

![VAE Encoder Diagram](/assets/images/diagram_vae_encoder.svg)

> **Note:** The VAE encoder is trained jointly with the rest of ACT using two loss terms: the standard action prediction loss, plus a **KL divergence** term that keeps the learned distribution close to a standard Gaussian. This prevents the distribution from collapsing to a single point (which would defeat the purpose of sampling).

---

## The Main Encoder

The main encoder takes the latent vector `z` produced by the VAE encoder and combines it with the current state information:

```
[z, joint states, image features]
```

This input is passed through **self-attention** and MLP layers — standard transformer blocks — to produce **contextualized representations** of the current state. "Contextualized" means that each element of the input has been allowed to attend to every other element: the image features can influence how joint states are interpreted, `z` can influence the representation of both, and so on.

The output of the main encoder is a rich, context-aware representation of the robot's current situation, which is then passed to the decoder.

---

## The Main Decoder

The main decoder takes a set of **query tokens** as input:
```
[query tokens]
```

Each query token corresponds to one future timestep in the predicted action chunk. If we want to predict a chunk of 10 future actions, we pass in 10 query tokens — one per predicted step. These tokens are learned positional embeddings that start with no task-specific information; their content is entirely shaped by what the decoder learns during training.

The decoder processes these queries in two stages:

**Self-attention** is applied first across the query tokens themselves. This ensures the predicted sequence is internally coherent — the action predicted at step 3 is aware of what was predicted at steps 1 and 2, so the chunk reads as one smooth, continuous motion rather than a series of disconnected predictions.

**Cross-attention** is then applied between the query tokens and the main encoder's output. This is the bridge between the two components: each query token reaches into the encoder's contextualised state representation — which encodes what the robot currently sees, its joint configuration, and the latent vector `z` — and uses that information to decide what action to predict at its timestep. Without this cross-attention step, the decoder would be predicting actions blind, with no knowledge of the current scene.

The decoder then outputs a predicted sequence of future joint positions — one per query token — which is compared against the ground truth actions from the demonstration. The resulting loss is propagated back through the decoder, main encoder, and VAE encoder via backpropagation to update all weights.

![ACT Architecture Overview](/assets/images/diagram_act_architecture.svg)

---

## Inference: What Happens at Test Time

At inference time — when the trained model is deployed on the real robot — the VAE encoder is **switched off entirely**. In its place, the latent vector `z` is fixed to the zero vector `z = 0`.

This works because the main encoder and decoder have been trained to operate robustly across a range of `z` values. 

The inference loop then runs as follows:

1. Cameras capture the current scene; the CNN extracts image features
2. Current joint states are read from the robot
3. The main encoder combines `[0, joint states, image features]` into a contextual representation
   > **Note:** Past actions are not passed to the main encoder — they were only ever an input to the VAE encoder during training. Since the VAE encoder is off at inference, past actions play no role.
4. The main decoder produces a chunk of predicted future actions
5. The robot executes the first action (or a weighted average of actions in the chunk)
6. The loop repeats at the next timestep

This approach — predict a chunk, execute the first step, predict again — allows the robot to continuously replan while still benefiting from the smoothness that chunking provides.

---

## Summary

### What Each Component Does

| Component | Input | Output | Training Only? |
|---|---|---|---|
| **CNN** | Raw camera images | Compact image feature vectors | No |
| **VAE Encoder** | CLS, joint states, image features, past actions | Latent vector `z` (sampled from μ, σ) | Yes |
| **Main Encoder** | `z`, joint states, image features | Contextualised state representation | No |
| **Main Decoder** | Query tokens + encoder output | Predicted sequence of future actions | No |

### Key Concepts

| Concept | What It Does |
|---|---|
| **Action Chunking** | Predicts multiple future actions at once for smooth, coherent motion |
| **Latent Distribution** | Introduces controlled variation during training to prevent overfitting |
| **CLS Token** | A single learned vector that summarises the entire input sequence |
| **Cross-Attention** | Allows predicted actions to be grounded in the current observed state |
| **VAE at Inference** | Replaced by `z = 0`; the model uses its "average" prediction |