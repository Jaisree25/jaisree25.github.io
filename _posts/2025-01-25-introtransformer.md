---
layout: post
title: "Intro to Transformers"
tags: [Transformers, Neural Networks, NLP, Attention]
---

---

## Table of Contents

1. [What Is a Transformer?](#what-is-a-transformer)
2. [RNNs vs Transformers](#rnns-vs-transformers)
3. [The Embedding Layer](#the-embedding-layer)
4. [The Attention Layer](#the-attention-layer)
5. [Queries, Keys, and Values](#queries-keys-and-values)
6. [Self-Attention vs Cross-Attention](#self-attention-vs-cross-attention)
7. [Multi-Head Attention](#multi-head-attention)
8. [The Full Forward Pass](#the-full-forward-pass)
9. [Summary](#summary)

---

## What Is a Transformer?

If you have been following developments in AI over the past few years, you have almost certainly encountered the word *transformer*. It is the architecture behind GPT, BERT, and virtually every modern large language model. But what exactly is a transformer, why is it so powerful, and how does it work?

This post builds directly on the neural networks guide. We have already seen how a vanilla neural network maps a fixed input to a fixed output — for instance, predicting whether a patient has diabetes from ten health metrics. The key limitation of that approach is that it produces a **single output for a single input**. Real language tasks — translation, text generation, summarization — require models that process and produce *sequences*. That need is what gave rise to transformers.

The transformer architecture was introduced in the 2017 paper *"Attention Is All You Need"* by Vaswani et al., and has since become the foundation of virtually every modern language model.

A transformer is a neural network designed to process and generate **sequences of information**. It is made up of three core components, repeated across multiple layers:

- **Embedding layer** — converts raw tokens (words or subwords) into dense numerical vectors
- **Attention layer** — allows each token to incorporate context from every other token in the sequence
- **MLP layer** — a standard feedforward network applied to each token's representation independently

Throughout this post, we will ground the explanation in a single task: **next-word prediction** — given a partial sentence, predict what word comes next.

---

## RNNs vs Transformers

Before transformers, the dominant architecture for sequence tasks was the **Recurrent Neural Network (RNN)**. Understanding where RNNs fell short is the clearest way to see what transformers solved.

An RNN processes a sequence **one token at a time**, left to right. Each step takes the current token and a hidden state carried forward from the previous step, and produces a new hidden state. This means the model must compress everything it has seen so far into a single vector before moving to the next token — a significant bottleneck for long sequences.

A transformer processes **all tokens in the sequence simultaneously**, in parallel. Rather than passing information through a sequential chain, every token attends directly to every other token in a single operation. This has two major consequences:

- **Speed** — parallel processing maps naturally to GPU hardware, making transformers dramatically faster to train than RNNs
- **Long-range dependencies** — a token at position 1 and a token at position 100 attend to each other directly, with no information loss from the intervening steps

![RNN vs Transformer Diagram](/assets/images/diagram_rnn_vs_transformer.svg)

> **Note:** RNNs process tokens sequentially, so training cannot be parallelized across the sequence. Transformers process all tokens at once, enabling efficient use of GPU parallelism.

---

## The Embedding Layer

Neural networks operate on numbers, not words. The embedding layer is responsible for converting each token into a numerical vector that the rest of the network can work with.

### Why Not Just Use Integer IDs?

Before the embedding layer can do anything, each word must be converted to an integer ID by mapping it against a vocabulary — a dictionary of every token the model knows. But raw integers have no useful geometry — the fact that "cat" is ID 1 and "mat" is ID 5 tells the model nothing about whether they are related.

### One-Hot Encoding: A Naive Approach

An early approach was **one-hot encoding**: represent each word as a vector of all zeros, with a single 1 at the index corresponding to that word's ID. With a vocabulary of 10,000 words, "cat" might be:

```
[0, 1, 0, 0, 0, ... 0]   ← 10,000 values, all zero except index 1
```

This has two serious problems. First, the vectors are enormous — 10,000 dimensions for every single word. Second, and more fundamentally, one-hot vectors carry no information about meaning. "Dog" and "puppy" are just as different from each other as "dog" and "democracy" — there is no similarity captured.

### Dense Embeddings

The embedding layer solves both problems by mapping each token ID to a **dense vector** — a short list of real numbers, typically 256 to 1024 values, learned during training.

```
one-hot:  [0, 0, 1, 0, 0, ... 0]           ← 10,000 dimensions, mostly zeros
embedding: [0.21, -0.54, 0.83, 0.12, ...]  ← 512 dimensions, all meaningful
```

These vectors are stored in an **embedding matrix** — one row per token in the vocabulary. Initially the values are random. As the model trains, the embedding matrix is updated via gradient descent alongside all other weights, and the vectors gradually adjust so that words used in similar contexts end up pointing in similar directions.

Similarity between two word vectors is measured using the **dot product** — a single number computed by multiplying corresponding elements and summing the results. A high dot product means the vectors are aligned, indicating the words are semantically related. This is why a well-trained embedding space places "dog" and "puppy" close together, and far from "democracy".

![Embedding Diagram](/assets/images/diagram_embedding.svg)

---

## The Attention Layer

The embedding layer encodes the *meaning* of individual words, but not their *context*. The word "model" means something very different in "fashion model" versus "neural network model" — yet both would receive the same embedding vector coming out of the embedding layer. The **attention layer** resolves this by updating each token's representation based on its relationship to all other tokens in the sequence.

### A Concrete Example

Take the sentence: **"There is a fluffy pink rock."**

The noun is "rock" and the adjectives are "fluffy" and "pink". The attention layer needs to figure out that "fluffy" and "pink" modify "rock" — and update the vector for "rock" so that it carries the meaning of "fluffy pink rock".

Here is how that happens.

### Step 1 — Attention Scores

For each token in the sequence, the model computes an **attention score** against every other token. These scores measure how relevant each token is to each other token. Higher score = more relevant.

Each token in the sequence has an embedding vector — a list of numbers representing its meaning. To compute how relevant any token is to "rock", we take the **dot product** of their two vectors:
```
attention_score(xᵢ, x_query) = xᵢ · x_query
```

For example, the embedding vector for "fluffy" points in a similar direction to the embedding vector for "rock" — both are associated with physical, tangible objects. Their dot product is therefore high (0.90). The embedding vector for "There" points in a very different direction — it is a filler word with no physical meaning — so its dot product with "rock" is low (0.01). We repeat this for every token in the sentence, giving us one raw score per token representing how relevant it is to "rock".

The raw attention scores for our sentence might look like this — the key insight is that "rock" has high scores with "fluffy" and "pink", since adjectives are strongly associated with the noun they modify:

```
         There   is     a    fluffy  pink   rock
There  [  0.00  0.00  0.00   0.00   0.00   0.00 ]
is     [  0.10  0.00  0.00   0.00   0.00   0.00 ]
a      [  0.05  0.10  0.00   0.00   0.00   0.00 ]
fluffy [  0.05  0.05  0.10   0.00   0.00   0.00 ]
pink   [  0.05  0.05  0.10   0.30   0.00   0.00 ]
rock   [  0.01  0.01  0.01   0.90   0.80   0.01 ]
                             ^^^^   ^^^^
```

> **Note:** The lower triangle is zeroed out — this is called **causal masking**, a technique that prevents earlier tokens from being influenced by later ones. It is used in GPT-style next-word prediction models. We cover this in depth in the LLM internals post.

### Step 2 — Softmax Normalization

Raw attention scores are converted into **attention weights** using the softmax function, so that the weights for each token sum to 1. This turns the scores into a proper probability distribution — how much attention should "rock" pay to each other word?

```
rock's attention weights (after softmax → sums to 1.0):

There: 0.01,  is: 0.01,  a: 0.01,  fluffy: 0.47,  pink: 0.42,  rock: 0.08
```

"fluffy" and "pink" dominate, which is exactly what we want.

### Step 3 — Weighted Sum (Context Vector)

Finally, the model computes a **context vector** for "rock" by taking a weighted sum of all token vectors, using the attention weights:

```
context("rock") = 0.01·v(There) + 0.01·v(is) + 0.01·v(a)
                + 0.47·v(fluffy) + 0.42·v(pink) + 0.08·v(rock)
```

This context vector is the new, enriched representation of "rock" — a blend of all tokens, dominated by "fluffy" and "pink". The same process runs for every token in the sequence, producing an updated representation for each one.

![Attention Score Diagram](/assets/images/diagram_attention_score.svg)

---

## Queries, Keys, and Values

The attention mechanism above computed scores by comparing token vectors directly against each other. In practice, transformers use a more powerful formulation that gives the model three separate learned projections per token — **Queries, Keys, and Values** — each with its own weight matrix learned during training.

- **Query (Q)** — *"What am I looking for?"* — the token asking the question
- **Key (K)** — *"What do I contain?"* — every token advertising its content
- **Value (V)** — *"What should I pass forward if selected?"* — the actual information to aggregate

For each input token embedding `x`, three vectors are computed:

```
Q = x · Wq
K = x · Wk
V = x · Wv
```

Where `Wq`, `Wk`, and `Wv` are learned weight matrices. The attention score between a query and all keys is computed as a dot product, scaled by √dₖ to prevent gradients from vanishing as vector dimensionality grows:

```
Attention(Q, K, V) = softmax( QKᵀ / √dₖ ) · V
```

The improvement over the simpler version is that the model now has three separate learned lenses for each token — it can learn *what to ask*, *what to advertise*, and *what to contribute* independently, all tuned by gradient descent.

---

## Self-Attention vs Cross-Attention

The attention mechanism described above is **self-attention**: the queries, keys, and values all come from the *same* sequence. Each token attends to every other token within the same input, learning how words within a sentence relate to one another.

**Cross-attention** is used when the model needs to align information from *two different sequences*. The queries come from one sequence, while the keys and values come from another. This is how encoder-decoder transformers connect understanding of the input to generation of the output — the decoder queries the encoder's representations using cross-attention at each generation step.

Common use cases for cross-attention:

- **Language translation** — decoder attends to the encoder's representation of the source sentence
- **Image captioning** — language decoder attends to visual features extracted from the image
- **ACT (robotics)** — the action decoder attends to the encoder's representation of the robot's current state

---

## Multi-Head Attention

A single attention operation produces one view of which tokens are relevant to which. But language contains many simultaneous relationships — syntactic, semantic, positional — and a single head cannot specialize in all of them at once.

**Multi-head attention** runs several attention operations in parallel, each with its own independent `Wq`, `Wk`, `Wv` weight matrices. Each **head** learns to focus on a different type of relationship. 

If the model dimension is 512 and there are 8 heads, each head operates in a 64-dimensional subspace — computationally similar to a single full-dimensional attention operation, but producing richer representations because the heads specialize independently.

In practice, one head might learn to track subject-verb relationships, another might connect pronouns to their referents, and another might attend to nearby positional context. We cannot tell the heads what to focus on — this specialization emerges entirely from training.

---

## The Full Forward Pass

Now we can put the full picture together. Given the partial sentence **"There was a"**, here is what happens at each stage:

### 1. Tokenization and Embedding

Each word is converted to a token ID, then looked up in the embedding matrix to produce a dense vector. Positional encodings are added to each vector to give the model information about the order of tokens — without this, the model would treat "cat sat on" and "on sat cat" identically.

### 2. Attention Layers

The token vectors pass through multiple stacked attention layers. At each layer, every token attends to every other token (subject to causal masking), and each token's vector is updated to reflect the context of the full sequence. After several layers, the representation of "a" has been enriched by its relationship to "There" and "was".

### 3. MLP Layers

After each attention layer, the updated vectors pass through a **feedforward MLP** applied independently to each token. The MLP refines each representation further — it is where much of the model's factual knowledge is believed to be stored.

### 4. Output

The final vector for the last token — "a" — is passed through a linear layer that projects it into the vocabulary space (one score per word in the vocabulary), followed by softmax to produce a probability distribution. The highest-probability word is the prediction.

```
Input:  "There was a"
Output probabilities: { "rock": 0.25, "dog": 0.10, "time": 0.18, ... }
Predicted next word:  "rock"
```

The predicted word is appended to the sequence, and the entire process repeats — this is how transformers generate text one token at a time.

![Transformer Forward Pass Diagram](/assets/images/diagram_transformer_forward_pass.svg)

> **Note:** Throughout this post, references to the model "learning" or "training" mean the same thing: weight matrices — the embedding matrix `WE`, query matrix `Wq`, key matrix `Wk`, value matrix `Wv`, and all MLP weights — being updated via gradient descent and backpropagation as described in the neural networks guide.

---

## Summary

### Core Components

| Component | What It Does |
|---|---|
| **Embedding layer** | Maps token IDs to dense vectors; learns semantic similarity |
| **Attention layer** | Updates each token's representation using context from all other tokens |
| **MLP layer** | Refines representations independently per token; stores factual knowledge |
| **Positional encoding** | Adds order information to token embeddings |

### Key Concepts

| Concept | One-Line Definition |
|---|---|
| **Dot product** | Measures similarity between two vectors — higher means more aligned |
| **Softmax** | Converts raw scores into weights that sum to 1 |
| **Context vector** | Weighted sum of all token vectors; the enriched output of attention |
| **Q, K, V** | Separate learned projections controlling what each token queries, advertises, and contributes |
| **Causal masking** | Prevents tokens from attending to future positions during next-word prediction |