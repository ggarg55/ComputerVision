# From RNNs to Attention: A Complete Tutorial

> **Learning Objectives**
> By the end of this tutorial, you will be able to:
> - Explain how bidirectional RNNs capture context from both directions
> - Evaluate language models using perplexity
> - Diagnose vanishing and exploding gradient problems
> - Understand how LSTM and GRU gates control memory
> - Build a sequence-to-sequence model for machine translation
> - Identify the bottleneck problem in encoder-decoder architectures
> - Implement dot-product attention to improve translation quality
> - Explain why transformers replaced RNNs

---

## Table of Contents

1. [The Big Picture](#1-the-big-picture)
2. [Recurrent Neural Networks and Bidirectional RNNs](#2-recurrent-neural-networks-and-bidirectional-rnns)
3. [Evaluating Language Models: Perplexity](#3-evaluating-language-models-perplexity)
4. [Training RNNs: Vanishing and Exploding Gradients](#4-training-rnns-vanishing-and-exploding-gradients)
5. [Gated Architectures: LSTM and GRU](#5-gated-architectures-lstm-and-gru)
6. [Sequence-to-Sequence Models](#6-sequence-to-sequence-models)
7. [The Bottleneck Problem](#7-the-bottleneck-problem)
8. [Attention Mechanism](#8-attention-mechanism)
9. [Transformers: Parallelization and Positional Encoding](#9-transformers-parallelization-and-positional-encoding)
10. [Practice Problems](#10-practice-problems)
11. [Summary](#11-summary)

---

## 1. The Big Picture

Imagine you are building a translator that also understands sentiment. You have a sentence in English:

> **"The movie was terribly exciting."**

You want to:
1. Classify whether the sentiment is positive or negative.
2. Translate the sentence into Hindi.
3. Generate a short headline.

All three tasks share a common foundation: **sequence modeling**. This tutorial follows one story:

```
RNN → Bidirectional RNN → Perplexity → Gradient Problems → LSTM/GRU
   → Seq2Seq → Bottleneck → Attention → Transformers
```

Each step solves a specific failure of the previous step.

---

## 2. Recurrent Neural Networks and Bidirectional RNNs

### 2.1 Standard RNN (Forward Only)

A standard RNN processes a sequence left-to-right. At each time step \(t\), it updates its hidden state:

\[
h_t = \tanh(W_{xh} x_t + W_{hh} h_{t-1} + b_h)
\]

For **next-word prediction**, the model only sees the past context:

\[
P(x_{t+1} \mid x_1, x_2, \dots, x_t)
\]

**Example:** In "The movie was terribly…", the forward RNN has seen "The movie was" and predicts the next word. It has **no idea** what comes after "terribly."

### 2.2 Bidirectional RNN

In many tasks — sentence classification, named entity recognition, sentiment analysis — the **entire sequence is available** at inference time. Why not use the right context too?

A bidirectional RNN runs two independent RNNs:

- **Forward RNN** → processes \(x_1, x_2, \dots, x_T\)
- **Backward RNN** → processes \(x_T, x_{T-1}, \dots, x_1\)

The final representation is the **concatenation** of both hidden states:

\[
H_t = [\overrightarrow{h_t}; \overleftarrow{h_t}]
\]

```
"The movie was terribly exciting."
     →→→→→→→→→→→→→→→→→→→→→→→→   (forward RNN)
                              ←←←←←←←←←←←←←←←←←←←←←←←←   (backward RNN)

For the word "terribly":
  Forward hidden state  = context from "The movie was"
  Backward hidden state = context from "exciting"
  Concatenation = context from BOTH directions
```

**Dimensionality:** If each direction produces a 300-dimensional vector, concatenation gives 600 dimensions. You can project it back to 300 with a learned weight matrix if needed.

### 2.3 Why This Matters

Consider the sentence: **"The movie was terribly exciting."**

- The word **"terribly"** is related to **"movie"** (what was terrible?)
- It is also related to **"exciting"** (the contrast creates meaning)

A forward RNN only knows the first relationship. A bidirectional RNN captures both.

> **Key Insight:** Bidirectional RNNs are not a new architecture — they are two standard RNNs sharing the same input but reading it in opposite directions. The only new operation is **concatenation** at each time step.

**Applications mentioned:** Sentence classification, named entity recognition, sentiment analysis — any task where the full sequence is available.

---

## 3. Evaluating Language Models: Perplexity

### 3.1 Definition

How do we know if a language model is good? The standard metric is **perplexity**.

Perplexity is the inverse probability of the corpus, normalized by the number of words:

\[
\text{Perplexity} = \exp\left( -\frac{1}{N} \sum_{t=1}^{N} \log P(x_{t+1} \mid x_1, \dots, x_t) \right)
\]

Equivalently:

\[
\text{Perplexity} = \exp(J(\theta))
\]

where \(J(\theta)\) is the **cross-entropy loss** — the same loss used during training.

### 3.2 Interpretation

**Lower perplexity is better.**

| Perplexity | Meaning |
|---|---|
| 1.44 | The model is roughly choosing among 1.44 possible words at each prediction |
| 4.07 | The model is more confused — choosing among ~4 probable words |

**Memory hook:** Perplexity is the **average size of the model's word-choice menu**. A smaller menu means the model is more confident.

### 3.3 Connection to Training

If the model assigns **high probability** to the correct word:
- Loss is low
- Perplexity is low

If the model assigns **low probability** to the correct word:
- Loss is high
- Perplexity is high

> **Key Insight:** Perplexity is not a separate metric from loss — it is simply the exponential of the cross-entropy loss. When you minimize loss, you minimize perplexity.

---

## 4. Training RNNs: Vanishing and Exploding Gradients

### 4.1 The Problem

Backpropagation through an RNN is a **long chain product**. To compute the gradient of the loss with respect to an early hidden state \(h_1\), you multiply the derivatives of all intermediate time steps:

\[
\frac{\partial L}{\partial h_1} = \frac{\partial L}{\partial h_T} \cdot \frac{\partial h_T}{\partial h_{T-1}} \cdots \frac{\partial h_2}{\partial h_1}
\]

If each derivative has magnitude **less than 1**, the product shrinks exponentially:

\[
0.5^{10} \approx 0.00098 \quad \text{(vanishing gradient)}
\]

If each derivative has magnitude **greater than 1**, the product explodes:

\[
2.5^{10} \approx 9536 \quad \text{(exploding gradient)}
\]

### 4.2 Consequences

- **Early layers learn very slowly** — the gradient signal never reaches them.
- **Training deep networks becomes very difficult.**
- **Long sequences may not learn at all** — the model effectively forgets the beginning of the sentence.

> **Memory hook:** Vanishing gradient is the **telephone game**. By the time the message reaches the first person, it has become a whisper.

### 4.3 Remedies

#### (a) Activation Functions

ReLU and similar activations can help. ReLU zeroes negative values and passes positive values linearly, which can reduce the derivative product.

#### (b) Weight Initialization

Choose good initial weights so the derivatives start in a reasonable range.

#### (c) Residual Connections

Pass the original input forward alongside the transformed output. If the transformation is close to identity, the original value is retained.

\[
h_t = f(h_{t-1}) + h_{t-1}
\]

#### (d) Gradient Clipping

For **exploding gradients**, limit the gradient magnitude to a threshold:

\[
g_{\text{new}} = \min\left(1, \frac{\text{max allowed gradient}}{\|g\|}\right) \cdot g
\]

**Example:** If the threshold is 10 and the gradient norm is 50, scale it down by \(10/50 = 0.2\).

**Important clarification:** Do not confuse **gradient** with **learning rate**. Gradient clipping limits the slope, not the step size.

> **Question:** Can activation functions act as gradient clipping?
>
> **Answer:** Yes, they can behave like clippers. But dedicated gradient-clipping algorithms exist because you can design the clipping function independently of the activation's own property. ReLU has its own behavior; clipping is a separate design choice.

#### (e) LSTM and GRU

The most important architectural improvement — discussed next.

---

## 5. Gated Architectures: LSTM and GRU

### 5.1 The Core Idea: Gates

A **gate** is a sigmoid layer:

\[
\sigma(z) = \frac{1}{1 + e^{-z}}
\]

Sigmoid maps any value to **(0, 1)**:

- Closer to **1** → keep/pass information
- Closer to **0** → drop/forget information

**Memory hook:** Gates are **bouncers at a club**. They decide who enters memory, who leaves memory, and who gets announced at the output.

### 5.2 LSTM (Long Short-Term Memory)

LSTM introduces three gates and a cell memory:

| Gate | Purpose |
|---|---|
| **Forget gate** \(f_t\) | What old information to forget |
| **Input gate** \(i_t\) | What new information to remember |
| **Output gate** \(o_t\) | What to output |

**Cell memory update:**

\[
C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t
\]

where \(\odot\) is **element-wise dot product** (because these are matrices).

**Hidden state:**

\[
h_t = o_t \odot \tanh(C_t)
\]

**Example:** Translating **"I like computer science"** to Hindi.

- The model may need to remember **"like"** but forget the exact English word order.
- The forget gate drops "the" and "was."
- The input gate keeps "like" and "computer."
- The output gate announces the Hindi translation at each step.

### 5.3 GRU (Gated Recurrent Unit)

GRU simplifies LSTM by using two gates:

- **Reset gate** — how much past information to forget
- **Update gate** — how much past information to carry forward

GRU has fewer parameters and often performs comparably to LSTM.

### 5.4 The Unified View

```
Standard RNN:        h_t = tanh(W x_t + U h_{t-1})
LSTM:                h_t = o_t ⊙ tanh(f_t ⊙ C_{t-1} + i_t ⊙ C̃_t)
GRU:                 h_t = (1 - z_t) ⊙ h_{t-1} + z_t ⊙ h̃_t
```

The same line of information flows through different gates, each extracting a different aspect:

- What to **forget**
- What to **learn**
- What to **output**

> **Key Insight:** You can design new gates for new architectures. LSTM and GRU were the most efficient designs discovered.

---

## 6. Sequence-to-Sequence Models

### 6.1 The Encoder-Decoder Architecture

Machine translation is the task of converting a sentence from one language to another. The **sequence-to-sequence (Seq2Seq)** model uses two RNNs:

- **Encoder RNN** — reads the source sentence and produces a context vector \(C\)
- **Decoder RNN** — generates the target sentence, initialized with \(C\)

```
Source: "I like computer science"
         ↓
   [Encoder RNN] → C (context vector)
         ↓
   [Decoder RNN] → "मुझे कंप्यूटर विज्ञान पसंद है"
```

### 6.2 The Decoder Equations

The encoder produces a final hidden state:

\[
C = h_T
\]

The decoder's initial hidden state is set to this context vector:

\[
S_0 = C
\]

At each decoding step:

\[
S_t = \tanh(W_d y_{t-1} + U_d S_{t-1})
\]

The first step unfolds as:

\[
S_1 = \tanh(W_d y_0 + U_d S_0)
\]

where \(y_0\) is the **start token**.

> **Key Insight:** The critical idea is **initializing \(S_0\) with the context vector \(C\)**. Without this, the two RNNs would operate independently and no translation would occur.

### 6.3 Start and End Tokens

- **Start token** — signals the decoder to begin generation
- **End token** — signals the decoder to stop

These are special tokens added to the vocabulary.

### 6.4 Training: End-to-End

The encoder and decoder are trained **together**, not separately:

- Compute loss with respect to target words: \(J_1, J_2, J_3, \dots\)
- Backpropagate through both the decoder and encoder
- Optimize as a single system

### 6.5 Seq2Seq is Versatile

Almost every text-generation problem can be written as **source → target**:

| Task | Source | Target |
|---|---|---|
| Translation | English sentence | Hindi sentence |
| Summarization | Long article | Short summary |
| Headline generation | News article | Headline |
| Dialogue | Conversation history | Next reply |
| Code generation | Natural language description | Code |

**Example (Dialogue):** Given a conversation between Agent 1 and Agent 2:

```
A1: "How are you?"
A2: "I'm fine, thanks."
A1: "What did you do today?"
```

Concatenate the first three turns as **source**. The target is Agent 2's next reply. Train a Seq2Seq model on this format.

### 6.6 Multi-Layer Deep Encoder-Decoder

Single-layer encoder-decoder models were not sufficient. Stacking more layers improved performance:

- BERT-style models: typically 12 layers
- Large language models: varying depths, still being experimented with

```
Input → [Encoder Layer 1] → [Encoder Layer 2] → ... → Context
                                                           ↓
Output ← [Decoder Layer 1] ← [Decoder Layer 2] ← ... ← Context
```

### 6.7 Decoder Direction

The decoder can be **unidirectional** or **bidirectional** — it is a design choice. For generation tasks, unidirectional is standard because the model should not see future target tokens.

---

## 7. The Bottleneck Problem

### 7.1 The Problem

Seq2Seq forces an entire sentence into **one fixed-size vector** \(C\).

If the source sentence is 100 words long, the encoder must compress all information into a single vector — and **not all words are equally important**.

**Memory hook:** Squeezing a 100-page book into one sticky note. You will lose details.

### 7.2 Consequences

- Important information gets lost.
- Translation quality degrades for long sentences.
- The decoder has no way to "look back" at specific source words.

### 7.3 The Fix: Attention

Instead of relying on one summary vector, let the decoder **look back at all encoder hidden states** and decide which ones are relevant at each decoding step.

---

## 8. Attention Mechanism

### 8.1 Intuition

When translating a sentence, a human translator constantly **glances back** at the source sentence — not just once at the beginning, but at every word they generate. Attention gives the decoder the same ability.

### 8.2 Computing Attention Scores

Given:

- Encoder hidden states: \(h_1, h_2, \dots, h_T\)
- Decoder hidden state at step \(t\): \(s_t\)

**Step 1 — Score:**

For **dot-product attention**:

\[
\text{score}(s_t, h_i) = s_t^\top h_i
\]

**Step 2 — Normalize with softmax:**

\[
\alpha_{ti} = \frac{\exp(\text{score}(s_t, h_i))}{\sum_{j=1}^{T} \exp(\text{score}(s_t, h_j))}
\]

**Step 3 — Weighted sum (context vector):**

\[
C_t = \sum_{i=1}^{T} \alpha_{ti} h_i
\]

### 8.3 Example: Translating "I like computer science"

**Encoder states** (dimension 2):

\[
h_1 = (2, 0), \quad h_2 = (0, -1), \quad h_3 = (2, 1), \quad h_4 = (0, 1)
\]

**Decoder state** at step 1:

\[
s_1 = (1, -2)
\]

**Dot products:**

\[
s_1^\top h_1 = 2, \quad s_1^\top h_2 = 2, \quad s_1^\top h_3 = 0, \quad s_1^\top h_4 = -2
\]

**Softmax** (assuming these are the logits):

\[
\alpha_{11} \approx 0.42, \quad \alpha_{12} \approx 0.42, \quad \alpha_{13} \approx 0.12, \quad \alpha_{14} \approx 0.04
\]

**Context vector:**

\[
C_1 = 0.42(2,0) + 0.42(0,-1) + 0.12(2,1) + 0.04(0,1) \approx (1.08, -0.30)
\]

This \(C_1\) is used to initialize or augment the decoder at step 1.

> **Memory hook:** Attention = the translator **keeps glancing back** at the source sentence instead of relying only on one summary note.

### 8.4 Where to Use the Context Vector

The attention context \(C_t\) can be used in several ways:

1. **Initialize the decoder** — replace \(S_0 = C\) with \(S_0 = C_{\text{attention}}\)
2. **Concatenate** with the decoder hidden state: \([s_t; C_t]\)
3. **Feed as input** to the next decoder step

The key idea is: **the better the context vector, the better the predictions.**

### 8.5 Attention Weights as Interpretability

Attention weights \(\alpha_{ti}\) tell you **which source words the model is focusing on** when generating each target word.

**Example — Sentiment Analysis:**

> "I am **happy** today."

When predicting the label **positive**, the attention weight on "happy" will be highest. This gives interpretability: we know *why* the model made its prediction.

**Example — Summarization:**

> Source: "Russia calls for joint front against **terrorism**."
> Target headline: "Russia, joint front, **terrorism**."

When generating "terrorism" in the headline, the attention weight on "terrorism" in the source is highest.

### 8.6 Practical Details

#### Padding

Sentences in a batch have different lengths. Add **pad tokens** to make them equal length.

#### Masking

Padded tokens should receive **zero attention**. Create a masking vector that sets attention scores for padded positions to \(-\infty\) before softmax (so they become 0 after softmax).

#### Number of Attention Scores

For each decoder step \(t\), you compute **one score per encoder hidden state**. If the input length is \(T\), you get \(T\) attention scores.

---

## 9. Transformers: Parallelization and Positional Encoding

### 9.1 The Fundamental Limitation of RNNs

In an RNN, \(h_t\) depends on \(h_{t-1}\). You **cannot compute all hidden states in parallel** — you must wait for the previous step.

This is called **lack of parallelizability**.

**Memory hook:** RNN = **whisper chain**. Each person must hear the previous person before passing the message.

### 9.2 Self-Attention: The Transformer Trick

Transformers use **self-attention** with **query, key, and value** vectors:

\[
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
\]

Because every token can attend to every other token **simultaneously**, all hidden representations can be computed **in parallel**.

**Memory hook:** Transformer = **group chat with timestamps**. Everyone talks at once, but timestamps record the order.

### 9.3 The Position Problem

If everything is computed in parallel, the model **loses word order**. In an RNN, position is encoded by the sequential recurrence. In a transformer, you must add **positional encoding** explicitly.

**Common approaches:**

- Sinusoidal positional encoding (original transformer)
- Learned positional embeddings
- **Rotary positional encoding** (RoPE) — used in modern LLMs

> **Key Insight:** Attention alone is not enough for sequence modeling. Positional encoding is what makes a transformer a *sequence* model.

### 9.4 What RNNs Had That Transformers Need to Recover

| RNN Property | Transformer Solution |
|---|---|
| Sequential order | Positional encoding |
| Local context | Self-attention (global context) |
| Parallelization (absent) | Self-attention (parallel by design) |
| Long-range dependency (weak) | Self-attention (direct connections) |

---

## 10. Practice Problems

### Problem 1: Bidirectional RNN Dimensions

A bidirectional RNN has forward and backward hidden states of dimension 300 each. What is the dimension of the concatenated hidden state? If you project it back to 300 dimensions, what weight matrix shape do you need?

<details>
<summary>Solution</summary>

- Concatenated dimension: \(300 + 300 = 600\)
- Projection matrix shape: \(600 \times 300\) (or \(300 \times 600\) depending on convention)

</details>

---

### Problem 2: Perplexity Interpretation

A language model has perplexity 1.44 on a test set. Another has perplexity 4.07. Which is better, and what does each value mean?

<details>
<summary>Solution</summary>

- **1.44 is better** (lower perplexity).
- 1.44: On average, the model is choosing among ~1.44 possible words at each prediction.
- 4.07: The model is more confused — choosing among ~4 probable words.

</details>

---

### Problem 3: Gradient Clipping

A gradient vector has norm 50. The clipping threshold is 10. What is the scaled gradient?

<details>
<summary>Solution</summary>

\[
g_{\text{new}} = \min\left(1, \frac{10}{50}\right) \cdot g = 0.2 \cdot g
\]

The gradient is scaled down by a factor of 0.2.

</details>

---

### Problem 4: LSTM Gates

Match each gate to its function:

| Gate | Function |
|---|---|
| Forget gate | ? |
| Input gate | ? |
| Output gate | ? |

<details>
<summary>Solution</summary>

- **Forget gate:** What old information to forget
- **Input gate:** What new information to remember
- **Output gate:** What to output

</details>

---

### Problem 5: Seq2Seq Initialization

Why is initializing the decoder hidden state \(S_0\) with the context vector \(C\) critical?

<details>
<summary>Solution</summary>

Without \(S_0 = C\), the encoder and decoder would operate independently. The decoder would have no information about the source sentence, and no translation would occur. The context vector is the **only channel** through which the source sentence influences the target generation.

</details>

---

### Problem 6: Dot-Product Attention

Encoder states (dimension 2):

\[
h_1 = (2, 0), \quad h_2 = (0, -1), \quad h_3 = (2, 1), \quad h_4 = (0, 1)
\]

Decoder state: \(s_1 = (1, -2)\).

Compute the attention scores (dot products) and identify which encoder state gets the highest raw score.

<details>
<summary>Solution</summary>

\[
s_1^\top h_1 = 2, \quad s_1^\top h_2 = 2, \quad s_1^\top h_3 = 0, \quad s_1^\top h_4 = -2
\]

Highest raw scores: \(h_1\) and \(h_2\) (tied at 2).

</details>

---

### Problem 7: Attention Context Vector

Using the scores from Problem 6, compute the softmax attention weights and the context vector \(C_1\).

<details>
<summary>Solution</summary>

**Softmax:**

\[
\alpha_{11} = \frac{e^2}{e^2 + e^2 + e^0 + e^{-2}} \approx 0.42
\]
\[
\alpha_{12} \approx 0.42, \quad \alpha_{13} \approx 0.12, \quad \alpha_{14} \approx 0.04
\]

**Context vector:**

\[
C_1 = 0.42(2,0) + 0.42(0,-1) + 0.12(2,1) + 0.04(0,1) \approx (1.08, -0.30)
\]

</details>

---

### Problem 8: Why Transformers?

Explain in one sentence why transformers replaced RNNs for most NLP tasks.

<details>
<summary>Solution</summary>

Transformers allow **parallel computation** of all token representations via self-attention, while RNNs must process tokens sequentially — and positional encoding solves the order problem that parallelism introduces.

</details>

---

### Problem 9: Masking

Why is masking necessary in attention for batched sequences?

<details>
<summary>Solution</summary>

Sentences in a batch have different lengths. Padding tokens are added to equalize lengths, but the model should not attend to padding. Masking sets attention scores for padded positions to \(-\infty\) before softmax, so they receive zero weight.

</details>

---

### Problem 10: Attention Interpretability

How can attention weights be used to interpret a sentiment classifier's prediction?

<details>
<summary>Solution</summary>

Attention weights show which input words the model focused on when making its prediction. For example, in "I am happy today," the attention weight on "happy" will be highest when predicting the positive label. This tells us *why* the model made its decision.

</details>

---

## 11. Summary

| Concept | Problem It Solves | Key Idea |
|---|---|---|
| **Bidirectional RNN** | Forward RNN misses right context | Two RNNs, concatenate hidden states |
| **Perplexity** | How to evaluate language models | Exponential of cross-entropy loss; lower is better |
| **Gradient clipping** | Exploding gradients | Scale gradient if norm exceeds threshold |
| **LSTM/GRU** | Vanishing gradients; uncontrolled memory | Sigmoid gates decide what to forget, keep, output |
| **Seq2Seq** | Machine translation | Encoder produces context \(C\); decoder starts with \(S_0 = C\) |
| **Attention** | Bottleneck of one context vector | Decoder looks back at all encoder states; weighted sum |
| **Transformer** | RNNs cannot parallelize | Self-attention + positional encoding |

### The One-Sentence Story

> An RNN reads left-to-right; a BiRNN reads both ways; gates control its memory; Seq2Seq translates by passing a context vector; attention lets the decoder look back; and transformers parallelize the whole process with positional encoding.

---

## Further Reading

- **Attention Is All You Need** (Vaswani et al., 2017) — the original transformer paper
- **Neural Machine Translation by Jointly Learning to Align and Translate** (Bahdanau et al., 2015) — the original attention paper
- **Long Short-Term Memory** (Hochreiter & Schmidhuber, 1997) — the LSTM paper
- **Sequence to Sequence Learning with Neural Networks** (Sutskever et al., 2014) — the Seq2Seq paper

---

*This tutorial is released under the MIT License. Contributions and corrections are welcome.*