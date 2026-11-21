# Neural Networks and Language Models: From Gradient Descent to RNNs

A hands-on tutorial covering neural network gradients, skip-gram embeddings, window-based language models, and recurrent neural networks — with worked examples, diagrams, and memory hooks.

---

## Table of Contents

1. [The Basic Neural Unit](#1-the-basic-neural-unit)
2. [Loss and Gradient for a Single Example](#2-loss-and-gradient-for-a-single-example)
3. [Gradient Descent: Why Subtract?](#3-gradient-descent-why-subtract)
4. [Skip-Gram with Negative Sampling](#4-skip-gram-with-negative-sampling)
5. [Window-Based Neural Language Model](#5-window-based-neural-language-model)
6. [RNN Language Model](#6-rnn-language-model)
7. [Training an RNN Language Model](#7-training-an-rnn-language-model)
8. [RNN Applications and Sequence Labeling](#8-rnn-applications-and-sequence-labeling)
9. [Counting Parameters](#9-counting-parameters)
10. [Practice Problems](#10-practice-problems)
11. [Key Takeaways](#11-key-takeaways)

---

## 1. The Basic Neural Unit

Linear regression predicts with \(y = mx + c\). A neural unit replaces that with a **nonlinear squashing function**:

\[
\hat{y} = \tanh(wx + b)
\]

- \(w\) — weight (a dial)
- \(b\) — bias (another dial)
- \(\tanh\) — squashes output into \((-1, 1)\)

> **Memory hook:** A neural unit is a machine with two dials. Turn the dials, change the prediction. \(\tanh\) keeps the output in a friendly range.

---

## 2. Loss and Gradient for a Single Example

### The loss

For a single training example with input \(x\) and true target \(y\), use **sum of squared errors**:

\[
L = (y - \hat{y})^2
\]

Substituting \(\hat{y} = \tanh(wx + b)\):

\[
L = \big(y - \tanh(wx + b)\big)^2
\]

### Worked example

Take \(x = 1\), \(y = 0.35\). Then:

\[
L = \big(0.35 - \tanh(w + b)\big)^2
\]

**Do not substitute \(w\) and \(b\).** They are the parameters we are learning. Keep them symbolic.

### Derivative with respect to \(w\)

Apply the chain rule in three layers:

1. **Outer square:** \(\dfrac{d}{d\hat{y}}(y-\hat{y})^2 = -2(y-\hat{y}) = 2(\hat{y}-y)\)
2. **Tanh derivative:** \(\dfrac{d}{dz}\tanh(z) = 1 - \tanh^2(z)\)
3. **Inside linear term:** \(\dfrac{\partial}{\partial w}(wx+b) = x\)

Multiply them:

\[
\boxed{
\frac{\partial L}{\partial w}
=
2\big(\tanh(wx+b)-y\big)
\left(1-\tanh^2(wx+b)\right)x
}
\]

At \(x = 1\), \(y = 0.35\):

\[
\frac{\partial L}{\partial w}
=
2\big(\tanh(w+b)-0.35\big)
\left(1-\tanh^2(w+b)\right)
\]

> **Memory hook:** “**2, tanh-minus-target, 1-minus-tanh-squared, x**.” Say it out loud once and it sticks.

---

## 3. Gradient Descent: Why Subtract?

The parameter update rule is:

\[
w \leftarrow w - \eta \frac{\partial L}{\partial w}
\]

where \(\eta\) is the learning rate.

### The compass analogy

- The gradient \(\partial L/\partial w\) points toward **steepest ascent** — uphill.
- We want to go **downhill**, so we walk the opposite direction: subtract the gradient.

### What if the gradient is negative?

If \(\partial L/\partial w < 0\), then:

\[
w - \eta(\text{negative}) = w + \text{positive}
\]

So \(w\) **increases**. A negative gradient pushes the weight up.

### Does a negative gradient guarantee the loss decreases?

**No.** It depends on the learning rate and the size of the gradient.

- A small step → loss decreases.
- A huge step → you can overshoot the valley and the loss can increase.

> **Memory hook:** Gradient is a compass pointing uphill. We walk the opposite way — but don’t run off a cliff.

---

## 4. Skip-Gram with Negative Sampling

### Configuration

A skip-gram Word2Vec model is trained with:

| Setting | Value |
|---|---|
| Vocabulary size | \(V\) |
| Embedding dimension | \(d\) |
| Context window size | 2 |
| Negative samples \(k\) | 10 |

### What window size = 2 means

Sentence: `cat sat on the mat`

Target word: `on`

Window size 2 means up to **2 words before and 2 words after**:

```
        ┌──── window ────┐
cat  sat  on  the  mat
     └── 2 ──┘  └── 2 ──┘
```

So the possible context words are: `cat`, `sat`, `the`, `mat` — at most **4 context words**.

### What negative sampling \(k = 10\) means

For each **positive** target–context pair, we draw **10 negative** context words from the vocabulary.

Total sigmoid computations per positive pair:

\[
1 \text{ positive} + 10 \text{ negatives} = 11
\]

With 4 context words:

\[
4 \times 11 = \boxed{44 \text{ sigmoid computations}}
\]

> **Memory hook:** One real friend, ten impostors, four friends around the target. \(4 \times 11 = 44\).

### Why sigmoid?

Sigmoid is used for **binary classification**: “Is this pair a real target–context pair, or a negative sample?” The model learns to say yes for positives and no for negatives.

### Embedding lookup = matrix row selection

Given a one-hot vector for the center word, multiplying by the embedding matrix \(W\) is equivalent to **selecting a row** of \(W\). It’s a lookup table.

Worked example:

\[
v_c = [0.4,\ 0.1]
\]

Then compute the score with an output vector \(u\):

\[
\text{score} = u^\top v_c
\]

Apply softmax over the vocabulary to get a probability distribution. In the worked example, the model predicts the **fourth word**.

### After one iteration

The weights have **not been updated** yet, so the final embedding for the center word is still:

\[
v_c = [0.4,\ 0.1]
\]

### Static embeddings

Word2Vec embeddings are **static**: the word “bank” has the same vector in “river bank” and “money bank.”

Contextual embeddings (transformers) solve this by producing different vectors for the same word in different contexts.

> **Memory hook:** Static embedding = one ID card per word, no matter where it appears. Contextual embedding = a new ID card for every sentence.

---

## 5. Window-Based Neural Language Model

### The task

Given a sequence of words \(x_1, x_2, \dots, x_t\), predict the next word.

Example task: **Named Entity Recognition (NER)** on the sentence “museums in Paris are amazing.” The word “Paris” should be tagged **location**.

### Architecture

```
x1  x2  x3  x4        (word embeddings or one-hot vectors)
 └──┴──┴──┘
     │
  concatenate / average / pool
     │
     E                 (fixed-size input representation)
     │
   h = f(WE + b)       (hidden layer)
     │
   U h + v             (output layer)
     │
   softmax
     │
   ŷ                   (probability over vocabulary)
```

- \(E\) — concatenated (or pooled) word embeddings
- \(W\) — input-to-hidden weights
- \(h\) — hidden representation
- \(U\) — hidden-to-output weights
- softmax — probability distribution over the vocabulary

### Problems with the fixed-window model

| Problem | What goes wrong |
|---|---|
| **Fixed window** | You must restrict the number of input words. You can never make the window large enough. |
| **Weight matrix grows** | Enlarging the window enlarges \(W\). |
| **No parameter sharing** | The same word “student” at position 2 vs. position 3 uses different weight matrices \(W_2, W_3\) — different internal representations. |
| **No symmetry** | Language does not care about absolute position. “The student opened there” and “Yesterday, students opened there” should be similar, but the model treats them differently. |
| **Poor generalization** | Because weights are not shared, learning does not transfer across positions. |

> **Memory hook:** A fixed-window NLM is a **photo with a fixed frame**. The same person in a different part of the photo gets a different label.

---

## 6. RNN Language Model

RNNs fix parameter sharing by reusing **the same weights at every time step**.

### Equations

For each time step \(t\):

\[
e_t = E x_t
\]

\[
h_t = \sigma(W_{xh} e_t + W_{hh} h_{t-1} + b_h)
\]

\[
\hat{y}_t = \text{softmax}(W_{out} h_t + b_{out})
\]

- \(E\) — embedding matrix
- \(W_{xh}\) — input-to-hidden weights
- \(W_{hh}\) — hidden-to-hidden weights (shared across time)
- \(W_{out}\) — hidden-to-output weights
- \(h_0\) — randomly initialized hidden state

### Unrolled diagram

```
x1 → e1 → h1 → ŷ1
          ↓
x2 → e2 → h2 → ŷ2
          ↓
x3 → e3 → h3 → ŷ3
          ↓
x4 → e4 → h4 → ŷ4  →  predict x5
```

The same \(W_{xh}\), \(W_{hh}\), and \(W_{out}\) are used at every time step. That is **parameter sharing**.

### Why this solves the window problem

- You can add as many time steps as you need → **any length input**.
- \(h_t\) carries information from previous steps → **information from many steps back**.
- Model size does **not** grow with input length.
- Same weights at every step → **symmetry** and **parameter sharing**.

### Disadvantages

- For long sequences, the recurrence and gradient computations become expensive.
- In practice, it is **hard to access information from many steps back** (vanishing-gradient intuition).

> **Memory hook:** RNN is a **rolling video with the same lens**. Same weights at every frame, remembers previous frames, but long videos get blurry.

---

## 7. Training an RNN Language Model

### Loss: cross-entropy / negative log likelihood

At every time step, compute:

\[
J_t = -\log p\big(y_t^{\text{gold}} \mid \text{previous words}\big)
\]

Total loss:

\[
J = \sum_t J_t
\]

### Worked example

Suppose the true next word is “students” and the model assigns it probability \(0.6\). Then:

\[
J_t = -\log(0.6)
\]

This is also called **negative log likelihood loss**.

### Teacher forcing vs. inference

| Mode | What is fed at the next time step |
|---|---|
| **Teacher forcing** (training) | The **ground-truth** previous word |
| **Inference** (testing) | The model’s **own prediction** |

Teacher forcing prevents early errors from propagating. Inference is how you actually generate text.

You can also **mix** teacher forcing and inference during training.

### Autoregressive generation

1. Start with a special **start token**.
2. Forward-propagate.
3. Sample a word from the probability distribution.
4. Feed that word as the input at the next time step.
5. Stop at an **end token** or a fixed length.

This is called **autoregressive generation** — the model consumes its own output.

```
[START] → “so” → “long” → “and” → “?” → [END]
```

> **Memory hook:** Teacher forcing = teacher gives the correct previous word. Inference = model eats its own output.

---

## 8. RNN Applications and Sequence Labeling

### Language modeling

Predict the next word at every time step. This is autoregressive generation.

### Sequence labeling

Label **every token** in the sequence. Examples:

- **POS tagging:** Janet/NNP Will/MD back/VERB the/DT bill/NN
- **NER:** “United Airlines” → organization

### BIO notation

To distinguish “beginning” from “inside” of an entity:

| Tag | Meaning |
|---|---|
| **B** | Beginning of an entity |
| **I** | Inside an entity |
| **O** | Outside any entity |

Example:

```
Jane    B-PER
Villanova  I-PER
works   O
at      O
United  B-ORG
Airlines  I-ORG
Holding   I-ORG
```

For \(n\) entity types, the number of output labels is:

\[
2n + 1
\]

The \(+1\) is the **O** tag.

With 5 entity types: \(2 \times 5 + 1 = 11\) output labels.

### Sequence classification

One label for the **whole sentence** (e.g., sentiment classification).

### Encoder–decoder

```
Input sentence → Encoder RNN → context vector → Decoder RNN → Output sequence
```

Used in machine translation, dialogue generation, and more.

> **Memory hook:** Label every word = sequence labeling. Label the whole sentence = sequence classification. Generate the next word = language modeling. Encode then decode = encoder–decoder.

---

## 9. Counting Parameters

**Rule:** Read the equation, find every weight matrix, multiply its dimensions, and add biases only if they appear in the equation.

### Example 1: RNN with input size 5, hidden size 6

| Matrix | Dimensions | Parameters |
|---|---|---|
| \(W_{xh}\) | \(5 \times 6\) | 30 |
| \(W_{hh}\) | \(6 \times 6\) | 36 |
| **Total** | | **66** |

If biases are counted, add hidden bias (6) and output bias (vocabulary size).

### Example 2: Output layer for sequence labeling

Hidden size 50, 11 BIO labels:

\[
W_{out}: 50 \times 11 = 550 \text{ parameters}
\]

Plus output bias: 11.

### Example 3: Embedding + RNN

Embedding dimension 300, hidden size 50, vocabulary size \(V\):

- Embedding matrix: \(V \times 300\)
- \(W_{xh}\): \(300 \times 50 = 15{,}000\)
- \(W_{hh}\): \(50 \times 50 = 2{,}500\)

> **Memory hook:** Parameters = every weight matrix in the equation. Multiply rows × columns. Add biases only if the equation has them.

---

## 10. Practice Problems

### Problem 1 — Gradient of squared error

For \(\hat{y} = \tanh(wx+b)\), \(L = (y-\hat{y})^2\), derive \(\partial L/\partial w\). Substitute \(x=1\), \(y=0.35\), but keep \(w,b\) symbolic.

**Answer:**

\[
\frac{\partial L}{\partial w}
=
2\big(\tanh(w+b)-0.35\big)
\left(1-\tanh^2(w+b)\right)
\]

### Problem 2 — Gradient sign

If \(\partial L/\partial w < 0\) and \(w \leftarrow w - \eta \partial L/\partial w\), does \(w\) increase or decrease? Does the loss always decrease?

**Answer:** \(w\) increases. Loss does **not** always decrease — it depends on the learning rate and gradient magnitude.

### Problem 3 — Skip-gram sigmoid count

Context window 2 on each side, negative sampling \(k=10\). How many sigmoid computations for one target-word occurrence with the full window?

**Answer:** 4 context words × (1 positive + 10 negatives) = **44**.

### Problem 4 — Parameter count

RNN with input size 5, hidden size 6. Count \(W_{xh}\) and \(W_{hh}\) parameters.

**Answer:** \(5\times6 + 6\times6 = 30 + 36 = 66\).

### Problem 5 — BIO labels

Five entity types. How many output labels in BIO notation?

**Answer:** \(2\times5 + 1 = 11\).

---

## 11. Key Takeaways

1. **Neural unit:** \(\hat{y} = \tanh(wx+b)\), trained with squared error and the chain rule.
2. **Gradient descent:** subtract the gradient because it points uphill. Negative gradient raises the weight. A negative gradient does not guarantee lower loss — the learning rate matters.
3. **Skip-gram:** one target, up to 4 context words (window 2), 10 negatives each → 44 sigmoids. Embeddings are static lookups.
4. **Window NLM:** fixed frame, no parameter sharing, poor generalization.
5. **RNN:** shared weights across time, any length input, but expensive and forgetful over long sequences.
6. **Training:** cross-entropy / negative log likelihood; teacher forcing for training, autoregressive generation for inference.
7. **Applications:** sequence labeling (POS, NER with BIO), sequence classification, encoder–decoder.
8. **Parameter counting:** multiply matrix dimensions; add biases only if present in the equation.

---

### One-sentence summary

> Use \(\tanh(wx+b)\), take squared error, backprop with the chain rule. Subtract the gradient to go downhill. In skip-gram, one target has four context words and ten negatives each — forty-four sigmoids. Fixed-window models are photos; RNNs are rolling videos with shared weights. Train with cross-entropy and teacher forcing; generate autoregressively. Count parameters by multiplying matrix dimensions, and remember BIO gives \(2n+1\) labels.