# Word Embeddings and Word2Vec: A Self-Contained Tutorial

This tutorial walks through the journey from **one-hot vectors** to **dense word embeddings** using **Word2Vec**. By the end, you will understand why one-hot representations fail, how context creates meaning, how Skip-gram and CBOW work, what softmax and likelihood mean here, and how to implement the core algorithm on a tiny corpus.

---

## 1. Why one-hot vectors fail

Suppose your vocabulary contains `hotel`, `motel`, `bank`, and many other words. A one-hot vector gives each word a unique position:

```text
hotel = [0, 0, 0, 0, 1, 0, 0, ...]
motel = [0, 0, 0, 0, 0, 0, 1, ...]
bank  = [0, 0, 1, 0, 0, 0, 0, ...]
```

Hotel and motel mean similar things. But if you compute cosine similarity or a dot product between their one-hot vectors, you get **0**. They are **orthogonal**. In fact, every pair of distinct one-hot vectors is orthogonal.

This creates several problems:

- There is **no natural notion of similarity** between one-hot vectors.
- All words are equally distant from each other.
- There is **no semantic structure**.
- The vectors are extremely **sparse**: one `1` and thousands or millions of zeros.
- Computation becomes inefficient.

Modern large models often use a vocabulary of roughly **250,000 tokens**—not millions or trillions. Parameter count is not vocabulary size. Even at 250k, a one-hot vector has length 250k. That is too large and too sparse for efficient computation.

You may already know dimensionality-reduction methods such as **PCA** and **LDA**. The goal here is similar in spirit but different in mechanism: convert sparse one-hot vectors into **dense vectors** that retain meaning while reducing size.

> **Key idea:** One-hot encoding knows a word’s identity, not its neighbours.

---

## 2. Meaning comes from context

Consider the word `bank`. It has multiple meanings:

- “I went to the **bank** to draw some money.” → financial bank
- “I sat near the river **bank**.” → river bank

The word alone is ambiguous. The **context** disambiguates it.

This is the distributional idea often summarized by J.R. Firth:

> “You shall know a word by the company it keeps.”

So instead of representing a word only by its one-hot identity, represent it by the words that frequently appear near it. A word’s meaning is stored in its context.

### Context window

When a word `w` appears in text, its context is the set of words that appear nearby. The number of words to consider on each side is the **window size**.

If window size is `m`, we look at up to `m` words before and `m` words after the target word. The window slides over the text.

Example:

```text
problems turning banking crisis
                 ↑
             center word
```

If `banking` is the center word and the window size is 2, the context words are drawn from the nearby words such as `problems`, `turning`, and `crisis`, depending on the exact window convention.

Diagram:

```text
Text:      ...  w1  w2  w3  w4  w5  w6  ...
Window:              [ w3  w4  w5 ]
Center:                   w4
Context:             w3        w5
```

The window slides one position at a time. Every word becomes a center word once. The first and last words naturally have fewer context words.

> **Key idea:** A word’s representation should come from its context, not from its one-hot identity alone.

---

## 3. Dense vectors and distributed representations

We want to replace a sparse one-hot vector with a compact dense vector:

```text
bank one-hot:  [0, 0, 1, 0, 0, 0, ...]        # sparse, length V
bank dense:    [0.21, -0.44, 0.87, 0.05, ...]  # dense, length D
```

Here `V` is vocabulary size and `D` is embedding dimension, such as 3, 100, or 300.

These dense vectors are called:

- **word vectors**
- **word embeddings**
- **neural word representations**

They fall under the general concept of **distributed representations**. “Distributed” means meaning is spread across many dimensions rather than assigned to one dimension per word.

The individual values are learned weights. They can be positive or negative. They are not usually interpretable one dimension at a time. The algorithm learns them.

A dense vector retains meaning while reducing size. That is the quality we want: **meaning is retained, but the vector is compact**.

---

## 4. Visualizing embeddings

High-dimensional vectors cannot be visualized directly. But you can apply dimensionality reduction, such as PCA or t-SNE, and plot them in 2D or 3D.

After plotting, similar words often form clusters. For example, words such as:

```text
come, go, keep, give, need, save, being, been, had, has, have
```

may appear close to each other.

Diagram:

```text
2D projection after dimensionality reduction

        come   go
          keep
     give      need
        save
              been
        has   had
```

This clustering is evidence that the vector representation captured semantic similarity. Similar words are close to each other.

> **Key idea:** If similar words end up close in vector space, the representation has learned something useful.

---

## 5. Word2Vec: Skip-gram and CBOW

Word2Vec is a framework for learning dense word vectors. It was introduced in a well-known paper by Mikolov et al. and released on arXiv. It is one of the first algorithms to make dense word representations practical and popular.

There are two main variants:

### Skip-gram

Given the **center word**, predict the **context words**.

```text
center word  →  context words
```

Example: given `banking`, predict nearby words such as `problems`, `turning`, `crisis`.

### CBOW

Given the **context words**, predict the **center word**.

```text
context words  →  center word
```

CBOW is the reverse of Skip-gram.

A word has two roles in Word2Vec:

- `v_w`: the vector for `w` when it is the **center** word.
- `u_w`: the vector for `w` when it is a **context** word.

This does **not** mean the word has two meanings. It means the algorithm uses two roles.

---

## 6. Skip-gram architecture

The Skip-gram model can be viewed as a small neural network:

```text
one-hot center word
        ↓
embedding matrix E  (V × D)
        ↓
center vector v_c   (D × 1)
        ↓
dot products with output vectors
        ↓
scores over vocabulary  (V × 1)
        ↓
softmax
        ↓
probability distribution over context words
```

More concretely:

1. The center word is represented as a one-hot vector of size `V`.
2. Multiplying by the embedding matrix selects the center word’s dense vector.
3. That center vector is compared with output/context vectors for every word in the vocabulary using dot products.
4. The dot-product scores are converted into probabilities with softmax.
5. The highest-probability word is the predicted context word.
6. Convert the predicted ID back to a word using `id_to_word`.

Diagram with shapes:

```text
E: V × D          embedding matrix
x: V × 1          one-hot center word
v_c = E^T x       D × 1 center vector

W: D × V          output/context weight matrix
scores = W^T v_c  V × 1
probs = softmax(scores)  V × 1
```

The output matrix `W` stores the context-word vectors. The embedding matrix `E` stores the center-word vectors.

Sometimes the embedding matrix is called a **lookup embedding** because selecting a word’s vector is just a lookup by ID.

---

## 7. The objective: likelihood and softmax

For a center word `c` and a context word `o`, we want to model:

```text
P(o | c)
```

Using dot-product similarity and softmax:

```text
P(o | c) = exp(u_o · v_c) / Σ_{w in vocab} exp(u_w · v_c)
```

- `u_o · v_c` is the similarity between the context word `o` and center word `c`.
- The numerator exponentiates that similarity.
- The denominator normalizes over the entire vocabulary.

Why exponentiate?

- Dot products can be negative.
- We want probabilities to be positive.
- Exponentiation guarantees positivity.
- Large similarities become much larger.
- Small similarities shrink rapidly.

The denominator is the normalization over the whole vocabulary. It turns scores into a probability distribution.

Example: `banking` and `crisis` should have a larger dot product than `banking` and `skeleton`. After exponentiation and normalization, `crisis` gets a larger probability.

### Likelihood over the corpus

For a corpus of `T` words and a fixed window size `M`, the likelihood is the product of the probabilities of context words given center words:

```text
L(θ) = Π_{t=1}^{T} Π_{j in window} P(w_{t+j} | w_t; θ)
```

Here `θ` are the model parameters—the embedding and output weight matrices.

A product of many probabilities is not numerically good. So we take the log:

```text
log L(θ) = Σ_{t=1}^{T} Σ_{j in window} log P(w_{t+j} | w_t; θ)
```

Maximizing this log-likelihood is equivalent to minimizing the negative log-likelihood. That is also equivalent to maximizing predictive accuracy.

This is a **maximum likelihood** view. The parameters responsible for the maximization are the weights `θ`.

> **Key idea:** Word2Vec is trained by making the observed context words more probable given their center words.

---

## 8. Softmax vs sigmoid

A common question: when do you use sigmoid and when do you use softmax?

- **Sigmoid** is used for binary classification.
- **Softmax** is used for multi-class classification.

In next-token prediction over a vocabulary, you need a probability for every word. If your vocabulary has 10 words, softmax gives a distribution over those 10 words with probabilities between 0 and 1 that sum to 1.

In Word2Vec, softmax is applied over the similarity scores to produce a probability distribution over the vocabulary.

---

## 9. Vector arithmetic and interpretation

If vectors capture meaning, arithmetic on vectors can capture relationships.

The classic example:

```text
king - man + woman ≈ queen
queen - female + male ≈ king
```

Diagram:

```text
king    - man    + woman  ≈ queen
queen   - female + male   ≈ king
```

The exact result depends on the learned vectors, but the idea is that semantic directions can be manipulated.

Vector arithmetic is also used in current AI-safety research. For example, if you have a **harmful prompt vector** and a **harmless prompt vector**, operations on them can help identify a direction in which a model should refuse or not refuse. This kind of operation is used to interpret the internal layers of large language models.

> **Key idea:** Dense word vectors support semantic arithmetic, and the same idea extends to interpreting modern LLMs.

---

## 10. How Word2Vec differs from generative models and attention

Word2Vec is not the same as an autoregressive generative model.

- A generative model typically uses the previous `n` words to predict the next word.
- Word2Vec uses **both before and after** context inside a window around a center word.

Attention also uses similarity, often cosine similarity, but Word2Vec does **not** compute attention scores. It directly uses dot-product scores to predict context words.

---

## 11. Implementation: Word2Vec on a tiny corpus

Use a very small corpus first. Debugging is much easier.

```text
cats like milk
cats drink milk
dogs like milk
```

### Step 1: Build the vocabulary

```text
vocab = [cats, like, milk, drink, dogs]
```

### Step 2: Build `word_to_id` and `id_to_word`

Assign each unique word a unique ID:

```text
word_to_id = {
  "cats": 0,
  "like": 1,
  "milk": 2,
  "drink": 3,
  "dogs": 4
}

id_to_word = {
  0: "cats",
  1: "like",
  2: "milk",
  3: "drink",
  4: "dogs"
}
```

You can reverse one dictionary to get the other, but keeping both is clear. IDs are unique, so there is no duplicate-word issue.

### Step 3: Create an embedding lookup

An embedding lookup maps each ID to a vector.

```text
embedding_lookup = {
  0: [0.2, -0.1, 0.4],
  1: [-0.3, 0.5, 0.1],
  ...
}
```

Initially, you can use one-hot vectors for practice. Later, switch to random dense vectors. The embedding lookup is easier to use in the training loop.

### Step 4: Create training pairs

Only pair a center word with **neighboring words inside the window**. Do not pair every word with every other word.

For `cats like milk` with window size 1:

```text
(cats, like)
(like, cats)
(like, milk)
(milk, like)
```

After converting words to IDs, the pairs look like:

```text
(0, 1)
(1, 0)
(1, 2)
(2, 1)
```

For a longer sentence with window size 2, the first and last words naturally have fewer context pairs.

Every word becomes a center word once. The first and last words will not pair with words that are outside the window.

### Step 5: Create the embedding and output matrices

Let `V` be vocabulary size and `D` be embedding dimension.

```text
E: V × D     input embedding matrix
W: D × V     output/context weight matrix
```

The one-hot center word is `V × 1`. The embedding lookup gives a `D × 1` center vector. Multiplying by the output matrix gives scores over the vocabulary. Softmax gives probabilities.

### Step 6: Forward pass

```text
center_id → v_c = E[center_id]        # D-dimensional vector
scores = W^T v_c                      # V scores
probs = softmax(scores)               # V probabilities
predicted_id = argmax(probs)
predicted_word = id_to_word[predicted_id]
```

### Step 7: Train

Use negative log-likelihood / cross-entropy as the loss. Update the embedding and output weights so that the observed context words become more probable.

Pseudocode:

```python
for epoch in range(num_epochs):
    for center_id, context_id in training_pairs:
        v_c = E[center_id]
        scores = W.T @ v_c
        probs = softmax(scores)
        loss = -log(probs[context_id])
        backprop_and_update(E, W, loss)
```

Use a small corpus while debugging. Increase the corpus later.

### Common pitfalls

- Do not pair the first word with the last word just because they are in the same sentence. Only neighboring words inside the window are paired.
- Do not skip `word_to_id` and `id_to_word`. You need IDs for matrix operations and words for interpretation.
- Do not start with a huge corpus. Start small, print the training pairs, and verify them.
- AI tools can help write code, but compile and understand the full Skip-gram implementation yourself.

---

## 12. Summary

```text
One-hot: sparse, orthogonal, no similarity.
Dense embedding: learned, compact, semantic.
Meaning = context; window = nearby words.
Skip-gram: center → context.
CBOW: context → center.
Score = dot product.
Probability = softmax.
Objective = maximize likelihood = minimize negative log-likelihood.
Vector arithmetic: king - man + woman ≈ queen.
Implementation: vocab → IDs → pairs → E/W → softmax → loss → update.
```

**Memory sentence:**  
> A word is known by the company it keeps; Word2Vec turns that company into a dense vector.