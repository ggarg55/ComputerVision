# From RNNs to Attention: A Practical Tutorial

**A standalone guide to recurrent models, sequence-to-sequence learning, and attention**

This tutorial explains how sequence models evolved from simple recurrent neural networks to attention-based models. It covers bidirectional RNNs, perplexity, vanishing and exploding gradients, gated RNNs, encoder-decoder models, neural machine translation, the bottleneck problem, attention, and the bridge to transformers.

**Prerequisites:** basic neural networks, activation functions, backpropagation, and cross-entropy loss.

**Running examples used throughout:**

- Sentiment/classification sentence: **“The movie was terribly exciting.”**
- Translation: **“I like computer science” → Hindi**
- Long document: **a 100-word review → one sentiment label or summary**
- Attention interpretation: **“I am happy today” → positive sentiment because of “happy”**

---

## Table of Contents

1. [Recurrent Neural Networks in One Minute](#1-recurrent-neural-networks-in-one-minute)
2. [Bidirectional RNNs: Two Readers, One Sentence](#2-bidirectional-rnns-two-readers-one-sentence)
3. [Perplexity: How Confused Is a Language Model?](#3-perplexity-how-confused-is-a-language-model)
4. [Vanishing and Exploding Gradients](#4-vanishing-and-exploding-gradients)
5. [Gated RNNs: LSTM and GRU](#5-gated-rnns-lstm-and-gru)
6. [Sequence-to-Sequence Models and Neural Machine Translation](#6-sequence-to-sequence-models-and-neural-machine-translation)
7. [The Bottleneck Problem](#7-the-bottleneck-problem)
8. [Attention: A Spotlight on the Right Input Words](#8-attention-a-spotlight-on-the-right-input-words)
9. [From Attention to Transformers](#9-from-attention-to-transformers)
10. [Summary, Exercises, and Glossary](#10-summary-exercises-and-glossary)

---

## 1. Recurrent Neural Networks in One Minute

A recurrent neural network processes a sequence one token at a time. At each step, it combines the current input with the previous hidden state:

\[
h_t = \tanh(W_{xh}x_t + W_{hh}h_{t-1} + b)
\]

The hidden state \(h_t\) is a summary of everything seen so far. This makes RNNs useful for language tasks such as next-word prediction and sentence classification.

But there is a key limitation: **RNNs are sequential**. To compute \(h_t\), you need \(h_{t-1}\). You cannot compute all hidden states in parallel. This will become important later when we compare RNNs with transformers.

**Memory hook:** An RNN is a reader who reads one word at a time and keeps a running notebook.

---

## 2. Bidirectional RNNs: Two Readers, One Sentence

A standard RNN reads left-to-right. A **bidirectional RNN** uses two independent RNNs:

- A **forward RNN** reads \(x_1 \to x_2 \to \dots \to x_T\).
- A **backward RNN** reads \(x_T \to x_{T-1} \to \dots \to x_1\).

At each token, the two hidden states are concatenated:

\[
h_t = [\overrightarrow{h_t}; \overleftarrow{h_t}]
\]

If each direction produces a 300-dimensional vector, the concatenated representation is 600-dimensional.

**Example:** In “The movie was terribly exciting”:

- The forward context around “terribly” sees “The movie was”.
- The backward context sees “exciting”.
- The concatenated representation captures that “terribly” relates both to “movie” and to “exciting”.

**Diagram:**

```text
Forward:   x1 → h1_f → h2_f → h3_f
Backward:  x1 ← h1_b ← h2_b ← h3_b
Combine:       [h1_f; h1_b]  [h2_f; h2_b]  [h3_f; h3_b]
```

Bidirectional RNNs are especially useful for **sentence classification**, where the whole sentence is available and the model can use both left and right context.

**Memory hook:** BiRNN = two readers reading the same sentence from opposite ends, then combining their notes.

---

## 3. Perplexity: How Confused Is a Language Model?

A standard metric for language models is **perplexity**. It measures how surprised or confused the model is when predicting the next word.

For a sequence \(w_1,\dots,w_N\):

\[
PP = \left(\frac{1}{P(w_1,\dots,w_N)}\right)^{1/N}
\]

Using the chain rule and next-word probabilities:

\[
PP = \exp\left(-\frac{1}{N}\sum_{t=1}^{N}\log P(w_t \mid w_{<t})\right)
\]

The term inside the exponential is the **cross-entropy loss** \(J(\theta)\). Therefore:

\[
PP = e^{J(\theta)}
\]

This is the same next-word probability used in the cross-entropy loss.

**Interpretation:**

- If the model assigns **high probability** to the correct word, the loss is low and perplexity is low.
- If the model assigns **low probability** to the correct word, the loss and perplexity are high.

**Examples:**

- \(PP = 1.44\): on average, the model is choosing among about **1.44 possible words** at each prediction. It is quite confident.
- \(PP = 4.07\): the model is more confused, effectively guessing among more probable words.

**Memory hook:** Perplexity is a “confusion dice.” Lower is better. \(PP=1.44\) is a nearly sure student; \(PP=4.07\) is a confused student.

---

## 4. Vanishing and Exploding Gradients

Training RNNs on long sequences is difficult because backpropagation goes through a long chain of time steps.

The loss with respect to an early hidden state becomes a product of many derivatives:

\[
\frac{\partial L}{\partial h_1}
=
\prod_{t=2}^{T}
\frac{\partial h_t}{\partial h_{t-1}}
\]

### Vanishing gradients

If each derivative is small, say \(0.5\), then after 10 steps:

\[
0.5^{10} \approx 0.001
\]

The gradient becomes extremely small. Early layers learn very slowly, and training deep networks becomes difficult.

**Consequences:**

- Early layers learn very slowly.
- Long-range dependencies are hard to learn.
- Deep recurrent networks become difficult to train.

### Exploding gradients

If the derivatives are large, say \(2.5\), then:

\[
2.5^{10} \approx 9536
\]

The gradient can grow extremely large, causing unstable training.

### Important distinction: gradient vs. learning rate

The **gradient** is the slope: how the loss changes with respect to weights.  
The **learning rate** is the step size used to update weights.

Do not confuse the two. A small gradient is not the same as a small learning rate.

### Common fixes

1. **ReLU activations**  
   ReLU maps negative values to zero and keeps positive values on a straight line. This can prevent long products from shrinking as badly.

2. **Suitable weight initialization**  
   Start with good initial weights rather than arbitrary ones.

3. **Residual connections**  
   Pass the original input forward so that if the change is small, the original information is retained.

4. **LSTM and GRU**  
   Gated architectures are the most important architectural improvement for recurrent models.

5. **Gradient clipping**  
   For exploding gradients, limit the gradient magnitude.

A common gradient-clipping rule is:

\[
g_{\text{clip}} =
\min\left(1,\frac{\tau}{\|g\|}\right)g
\]

where \(\tau\) is a threshold such as 10 or 5. This stabilizes the learning process.

**FAQ:** Can activation functions act as gradient clipping?  
Yes, activation functions can behave like clippers, but dedicated gradient-clipping algorithms are more explicit and controllable. ReLU has its own behavior; clipping is a separate mechanism designed specifically to limit gradient size.

**Memory hook:** A long chain of whispers either fades away (vanishing) or becomes a shout (exploding). Gradient clipping is the volume limiter.

---

## 5. Gated RNNs: LSTM and GRU

When processing a long sequence, not every piece of information is useful. To predict a later token, you may not need information from every earlier token. The model needs a way to **remember**, **forget**, and **output** selectively.

Gated RNNs introduce **gates**. A gate uses a sigmoid:

\[
\sigma(z) \in (0,1)
\]

If the value is near 1, more information passes through. If it is near 0, information is suppressed.

### LSTM gates

LSTM introduces three gates:

- **Forget gate** \(f_t\): what old cell memory to forget?
- **Input gate** \(i_t\): what new information to remember?
- **Output gate** \(o_t\): what exactly to output?

A simplified LSTM cell memory update is:

\[
C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t
\]

\[
h_t = o_t \odot \tanh(C_t)
\]

Here \(\odot\) denotes **element-wise multiplication**.

The idea is:

- \(f_t\) controls when to forget old information.
- \(i_t\) controls when to remember new information.
- \(o_t\) controls what to output.

### GRU

GRU uses **reset** and **update** gates instead of three LSTM gates. It is a simpler gated architecture, but the core idea is the same: learn what to keep, what to discard, and what to output.

**Memory hook:** Gates are bouncers at an information door. The forget gate throws out old guests; the input gate checks new guests; the output gate decides who speaks.

---

## 6. Sequence-to-Sequence Models and Neural Machine Translation

Machine translation means translating a sentence from one language to another. A **sequence-to-sequence (seq2seq)** model uses two RNNs:

- An **encoder RNN** reads the source sentence.
- A **decoder RNN** generates the target sentence.

### Encoder

Given source tokens \(x_1,\dots,x_T\):

\[
h_t = \tanh(W_e x_t + U_e h_{t-1})
\]

The final encoder hidden state is the **context vector**:

\[
C_T = h_T
\]

This vector is meant to capture the meaning of the source sentence.

### Decoder

The decoder generates target tokens \(y_1,\dots,y_T\). Its first hidden state is initialized with the encoder context:

\[
s_0 = C_T
\]

At each step:

\[
s_t = \tanh(W_d y_{t-1} + U_d s_{t-1})
\]

A **start token** begins generation. Special tokens such as start, EOS, and padding are used to manage generation and variable-length sequences.

**Diagram:**

```text
Source:   x1 → x2 → x3 → [Encoder RNN] → C
Target:   <START> → y1 → y2 → y3 → <EOS>
                         ↑
                         |
                   [Decoder RNN] initialized with C
```

The key idea is that the decoder starts from the encoder’s representation. If you simply had two RNNs and never passed the encoder representation to the decoder, there would be no translation signal. Initializing \(s_0\) with \(C_T\) is the central trick.

### Training

Training compares decoder outputs with ground-truth target words and computes losses:

\[
J_1, J_2, J_3, \dots
\]

The whole encoder-decoder system is trained **end to end** as a single system, not encoder first and decoder later.

### Multi-layer deep encoder-decoder

One layer is often not sufficient. We can stack more encoder and decoder layers. The final representation from one layer can be passed to the next. Through experimentation, deeper models often work better. For example, BERT-style models commonly use 12 layers, while large language models vary.

### Why seq2seq is transformative

Almost any text problem can be written as a **source sentence → target sentence** problem:

- **Summarization:** long text → summary
- **Headline generation:** news article → headline
- **Dialogue:** \(A_1, A_2, A_1\) → reply of \(A_2\)
- **Parsing**
- **Code generation**
- **Question answering**
- **Text generation in general**

For dialogue, concatenate the previous conversation turns into a source sequence and use the desired reply as the target.

**Memory hook:** Encoder = a reader who writes one summary note. Decoder = a writer who starts from that note and writes the translation.

---

## 7. The Bottleneck Problem

In plain seq2seq, the entire source sentence is compressed into one context vector \(C_T\).

For a 100-word sentence, squeezing all information into one small vector can lose important information. Not all words are equally important, and the decoder may need different parts of the input at different decoding steps.

This is the **bottleneck problem**.

**Intuition:** Imagine summarizing a 100-word review into one short note and then writing a detailed response using only that note. Some important details may be lost.

Attention solves this by letting the decoder look back at all encoder hidden states and decide which input words matter at each decoding step.

---

## 8. Attention: A Spotlight on the Right Input Words

Attention gives the decoder direct access to all encoder hidden states.

Given:

- Decoder hidden state \(s_t\)
- Encoder hidden states \(h_1,\dots,h_T\)

### Step 1: Compute attention scores

For **dot-product attention**:

\[
e_{t,i} = s_t^T h_i
\]

This score measures how relevant encoder state \(h_i\) is to decoder state \(s_t\).

### Step 2: Normalize with softmax

\[
\alpha_{t,i}
=
\frac{\exp(e_{t,i})}{\sum_j \exp(e_{t,j})}
\]

The attention weights \(\alpha_{t,i}\) are between 0 and 1 and sum to 1.

### Step 3: Compute the context vector

\[
c_t = \sum_i \alpha_{t,i} h_i
\]

This weighted context vector is used by the decoder, either as input, as initialization, or concatenated with the decoder state.

**Diagram:**

```text
s_t · h1 → e_t1 → softmax → α_t1
s_t · h2 → e_t2 → softmax → α_t2
s_t · h3 → e_t3 → softmax → α_t3
...
c_t = Σ α_ti h_i
```

At a given decoding step \(t\), \(t\) is fixed and \(i\) moves over all encoder states. If the input length is 3, you get three attention scores: \(\alpha_{t1}, \alpha_{t2}, \alpha_{t3}\).

### Numerical example

Suppose the encoder produces four hidden states of dimension 2:

\[
h_1=(2,0),\quad h_2=(-1,2),\quad h_3=(1,-1),\quad h_4=(0,1)
\]

At a particular decoding step, the decoder state is:

\[
s=(1,-2)
\]

Compute dot products:

\[
s\cdot h_1 = 2,\quad
s\cdot h_2 = -5,\quad
s\cdot h_3 = 3,\quad
s\cdot h_4 = -2
\]

Apply softmax:

\[
\alpha \approx [0.268,\ 0.0002,\ 0.727,\ 0.005]
\]

The third encoder state gets the most attention.

The context vector is:

\[
c
=
0.268(2,0)
+
0.0002(-1,2)
+
0.727(1,-1)
+
0.005(0,1)
\approx
(1.26,-0.72)
\]

If the context vector is not explicitly given, you can use the weighted sum \(c_t=\sum_i \alpha_{t,i}h_i\), or concatenate all encoder hidden states.

### Padding and masking

Variable-length sentences require **padding**. We add special tokens such as:

- **start token**
- **EOS token**
- **pad token**

Padding makes sequences equal length. However, the model should not pay attention to padded tokens. **Masking** gives padded positions zero attention.

Transformers also have length limits and require padding, although they handle sequences differently from RNNs.

### Interpretability

Attention can be visualized as an **attention plot**: input words on one axis, output words on another. Lighter colors indicate more importance.

Examples:

- In translation, when predicting a word related to “economic zone” or “European”, attention shifts to the relevant source words.
- In sentiment analysis, for “I am happy today”, the word “happy” receives high attention when predicting a positive label.
- In summarization, for a source sentence containing “Russia calls for joint front against terrorism”, when predicting “terrorism” in the headline, attention focuses on the word “terrorism” in the source.

Attention therefore improves neural machine translation performance and provides a way to interpret what the model is focusing on.

**Memory hook:** Attention is a spotlight. The decoder asks, “Which source words should I look at for this output word?”

---

## 9. From Attention to Transformers

Recurrent models have two major limitations:

1. **Lack of parallelizability**  
   To compute \(h_t\), you need \(h_{t-1}\). You cannot compute all positions in parallel.

2. **Long-range dependencies are hard to learn**  
   Information must pass through many recurrent steps.

Self-attention addresses the parallelization problem. In self-attention, each token computes **Query, Key, and Value** vectors. Because these computations do not depend on a sequential recurrence, hidden representations for all input tokens can be computed in parallel.

But there is a catch: if everything is computed in parallel, the model loses word order. The question is: **what makes a sequence a sequence?** The answer is **position**.

RNNs have order built into the recurrence. Transformers add **positional encoding** explicitly. Positional encoding is essential. Modern systems often use improvements such as **rotary positional encoding**; simple sine/cosine positional encoding is not the modern default.

The transformer architecture relies on self-attention and positional encoding. This is the bridge from recurrent sequence models to modern large language models.

**Memory hook:** Self-attention is a parallel committee. Positional encoding gives each committee member a badge showing where they sat in the sentence.

---

## 10. Summary, Exercises, and Glossary

### One-page summary

| Concept | Simple memory |
|---|---|
| RNN | Reads one token at a time; sequential |
| BiRNN | Two readers, left-to-right and right-to-left; concatenate notes |
| Perplexity | Confusion dice; lower is better; equals \(e^{\text{loss}}\) |
| Vanishing gradient | Whisper dies over a long chain; \(0.5^{10}\) is tiny |
| Exploding gradient | Whisper becomes a shout; clip it |
| Gradient clipping | Volume limiter: \(\min(1,\tau/\|g\|)g\) |
| LSTM/GRU | Bouncers: forget, input, output; reset/update |
| Seq2seq | Encoder reads, decoder writes; start decoder with context |
| Bottleneck | 100 words squeezed into one note loses information |
| Attention | Spotlight on relevant input words; weighted sum of encoder states |
| Self-attention | Parallel committee; add position badges |
| Positional encoding | Order is lost in parallel; add position information |

### Exercises

1. **BiRNN dimensions**  
   A forward RNN produces 300-dimensional hidden states and a backward RNN produces 300-dimensional hidden states. What is the dimension of the concatenated BiRNN state?

2. **Perplexity**  
   A language model has cross-entropy loss \(J=0.36\). Compute its perplexity. Is the model confident or confused?

3. **Gradient intuition**  
   Explain why \(0.5^{10}\) becomes very small and why this causes vanishing gradients in RNNs.

4. **Gradient clipping**  
   Given gradient \(g=(3,4)\) and threshold \(\tau=2\), compute \(\|g\|\) and the clipped gradient.

5. **LSTM gates**  
   In one sentence each, explain the roles of the forget gate, input gate, and output gate.

6. **Seq2seq**  
   Describe how you would frame dialogue reply generation as a seq2seq problem.

7. **Attention numerical**  
   Repeat the attention example with \(s=(2,-1)\) and the same four encoder states. Compute the dot products, softmax weights, and context vector.

8. **Padding and masking**  
   Why is masking necessary when using padded sequences?

9. **Attention interpretability**  
   For “I am happy today”, which word would you expect to receive high attention when predicting a positive sentiment label? Why?

10. **Parameter counting**  
    For an attention layer, identify which weight matrices are learned and count the parameters for a small example.

### Glossary

- **Hidden state:** A vector summarizing the sequence seen so far.
- **Context vector:** A representation of the source sentence passed from encoder to decoder.
- **Perplexity:** A language-model metric equal to the exponential of cross-entropy loss.
- **Vanishing gradient:** Gradients become extremely small during backpropagation.
- **Exploding gradient:** Gradients become extremely large during backpropagation.
- **Gradient clipping:** Limiting gradient magnitude to stabilize training.
- **Gate:** A sigmoid-controlled mechanism that decides how much information to pass.
- **Cell memory:** The LSTM’s internal memory state.
- **Encoder:** The RNN that reads the source sentence.
- **Decoder:** The RNN that generates the target sentence.
- **Start token:** A special token that begins generation.
- **EOS token:** A special token that marks the end of a sequence.
- **Padding:** Adding special tokens to make sequences equal length.
- **Masking:** Preventing attention from using padded positions.
- **Attention score:** A measure of relevance between a decoder state and an encoder state.
- **Softmax:** A normalization function that turns scores into probabilities.
- **Self-attention:** Attention where tokens attend to other tokens in the same sequence.
- **Positional encoding:** Information added to tokens so the model knows their order.

---

This tutorial is self-contained and suitable for open-source distribution. It covers the full path from recurrent models to attention and prepares learners for transformer architectures.