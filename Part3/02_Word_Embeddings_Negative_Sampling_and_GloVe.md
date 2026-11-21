# Word Embeddings, Negative Sampling, and GloVe: A Complete Tutorial

> **What you'll learn:** Why Word2Vec is slow, how negative sampling fixes it, why co-occurrence matrices are sparse, and how GloVe turns co-occurrence ratios into dense word vectors. By the end, you'll understand the math, the intuition, and how to implement both methods.

---

## Table of Contents

1. [The Problem: Why Word2Vec Hurts at Scale](#1-the-problem-why-word2vec-hurts-at-scale)
2. [Negative Sampling: Sample the Few, Not the All](#2-negative-sampling-sample-the-few-not-the-all)
3. [The Unigram Trap: Why "the" Ruins Everything](#3-the-unigram-trap-why-the-ruins-everything)
4. [Co-occurrence Matrices: Counting Is Cheap, Storing Is Not](#4-co-occurrence-matrices-counting-is-cheap-storing-is-not)
5. [GloVe: Meaning Lives in Ratios](#5-glove-meaning-lives-in-ratios)
6. [The GloVe Loss Function](#6-the-glove-loss-function)
7. [Implementing Negative Sampling](#7-implementing-negative-sampling)
8. [Implementing GloVe](#8-implementing-glove)
9. [What to Expect When You Plot Embeddings](#9-what-to-expect-when-you-plot-embeddings)
10. [Preprocessing: Clean Data Beats Clever Models](#10-preprocessing-clean-data-beats-clever-models)
11. [Edge Cases and Practical Warnings](#11-edge-cases-and-practical-warnings)
12. [Where This Goes Next](#12-where-this-goes-next)
13. [One-Page Cheat Sheet](#13-one-page-cheat-sheet)

---

## 1. The Problem: Why Word2Vec Hurts at Scale

Word2Vec learns word vectors by predicting context words from a center word. For a center word \(c\), the model computes a probability distribution over the **entire vocabulary**:

\[
P(w \mid c) = \frac{\exp(v_c \cdot u_w)}{\sum_{w' \in V} \exp(v_c \cdot u_{w'})}
\]

The denominator sums over **every word in the vocabulary** — often **more than 100,000 words**.

### A simple analogy

Imagine a shopkeeper who, for every single sale, must check **every item in a 100,000-item warehouse** to compute a probability. That's Word2Vec's denominator. It's correct, but painfully slow.

**The cost:** For each center word, you compute similarity with all 100,000 words. Multiply that by millions of training pairs, and training becomes computationally expensive.

**The question:** Can we avoid touching the whole vocabulary while still learning good embeddings?

**The answer:** Yes — **negative sampling**.

---

## 2. Negative Sampling: Sample the Few, Not the All

### The core idea

Instead of predicting which word from the full vocabulary appears in context, train a **binary logistic regression** classifier to distinguish:

- **Real (positive) pairs** — words that actually appear together
- **Noisy (negative) pairs** — random words that don't

> **Memory hook:** *"Sample the few, not the all."*

### Running example

Take the sentence:

> **"the cat sat on the mat"**

For center word **`cat`**:

| Pair type | Words | What the model learns |
|---|---|---|
| **Positive** | `sat`, `on` | Pull these **closer** to `cat` |
| **Negative** | `apple`, `car`, `quickly` | Push these **farther** from `cat` |

The model doesn't need to know about all 100,000 words. It only needs to get **one real pair right** and **a few fake pairs wrong**.

### The loss function

For a center word \(c\) and a true context word \(w\):

\[
L_{\text{pos}} = -\log \sigma(v_c \cdot u_w)
\]

For each negative sample \(w'\):

\[
L_{\text{neg}} = -\log \sigma(-v_c \cdot u_{w'})
\]

Total loss for one positive pair with \(K\) negatives:

\[
L = -\log \sigma(v_c \cdot u_w) - \sum_{k=1}^{K} \log \sigma(-v_c \cdot u_{w_k})
\]

Where \(\sigma(x) = \frac{1}{1 + e^{-x}}\) is the **sigmoid function**, squashing any number into \((0, 1)\).

### Reading the two terms

**Positive term:** \(-\log \sigma(v_c \cdot u_w)\)

- If `king` and `queen` frequently co-occur, the model increases their dot product.
- A larger dot product → larger sigmoid → smaller loss.
- **Effect:** true pairs get pulled together.

**Negative term:** \(-\sum \log \sigma(-v_c \cdot u_{w'})\)

- For `king` with `table`, `banana`, `car`, the model wants \(v_c \cdot u_{w'}\) to be **small** (even negative).
- Small dot product → \(\sigma(-x)\) is large → loss is small.
- **Effect:** fake pairs get pushed apart.

### Maximization vs. minimization

You'll see this written both ways. They're equivalent:

**Maximization form** (maximize probability of true pairs, minimize probability of random pairs):

\[
\max \left[ \log \sigma(v_c \cdot u_w) + \sum_{k=1}^{K} \log \sigma(-v_c \cdot u_{w_k}) \right]
\]

**Minimization form** (minimize the negative of the above):

\[
\min \left[ -\log \sigma(v_c \cdot u_w) - \sum_{k=1}^{K} \log \sigma(-v_c \cdot u_{w_k}) \right]
\]

The negative sign in front of the negative-sample term is what makes the "push apart" interpretation correct. Without it, you'd be maximizing the probability of random words appearing — the opposite of what you want.

### Why this is faster

| Method | Comparisons per center word |
|---|---|
| Full softmax | \(V\) (e.g., 100,000) |
| Negative sampling | \(1 + K\) (e.g., 6) |

With \(K = 5\), you go from 100,000 comparisons to **6**. That's a ~16,000× reduction per training step.

---

## 3. The Unigram Trap: Why "the" Ruins Everything

### The naive approach

To sample negatives, the obvious choice is to draw from the **unigram distribution** \(U(w)\) — the frequency of each word in the corpus.

### The problem

Words like **`the`**, **`of`**, **`a`**, **`is`**, **`on`** are extremely frequent. If you sample from raw unigram frequency, you'll mostly get **stop words**.

**Example:** If 8% of tokens are `the`, then roughly 8 out of every 100 negative samples will be `the`. The model spends most of its negative-sampling budget learning that `the` is not a good context for everything — which it already knows. You waste computation and learn little.

### The fix: reshape the distribution

Instead of sampling directly from \(U(w)\), use a **reshaped distribution**. A common choice is:

\[
P_{\text{neg}}(w) \propto U(w)^{3/4}
\]

This dampens the frequency of very common words and boosts the relative probability of rare words.

**Before reshaping:** `the` might be 8% of samples.  
**After reshaping:** `the` might be 2% of samples, while rare-but-meaningful words get a fairer chance.

> **Key takeaway:** The exact reshaping method is a design choice. The important thing is to **be aware of the problem** and choose a sampling method that solves it. You can use \(U(w)^{3/4}\), uniform sampling over the vocabulary, or any method that prevents stop words from dominating.

### Are stop words positive or negative?

A common confusion: if `the` appears in a sentence, is `(cat, the)` a positive or negative pair?

**Answer:** If it appears in the actual sentence, it's a **positive pair**. The model should learn that `the` can appear near `cat`.

**But in practice:** You **preprocess** the data and remove stop words before training. This is why preprocessing matters — if you leave stop words in, the model will learn mostly about stop words. Clean data produces clean embeddings.

---

## 4. Co-occurrence Matrices: Counting Is Cheap, Storing Is Not

### The motivation

Word2Vec slides a window through the corpus, treating every word as a center word and its neighbors as context. This means traversing the entire corpus **many times**.

**The question:** Instead of repeatedly sliding a window, why not just **accumulate statistics** — count how often words appear near each other?

**The answer:** Build a **co-occurrence matrix**.

### What is a co-occurrence matrix?

A matrix \(X\) where:

\[
X_{ij} = \text{number of times word } j \text{ appears in the context of word } i
\]

### Running example

Corpus:

> `I like deep learning.`  
> `I like NLP.`  
> `I enjoy flying.`

**Step 1: Build the vocabulary**

Unique words: `I, like, deep, learning, NLP, enjoy, flying`

**Step 2: Count co-occurrences** (with a small window, say ±2 words)

|  | I | like | deep | learning | NLP | enjoy | flying |
|---|---|---|---|---|---|---|---|
| **I** | 0 | 2 | 0 | 0 | 0 | 1 | 0 |
| **like** | 2 | 0 | 1 | 0 | 1 | 0 | 0 |
| **deep** | 0 | 1 | 0 | 1 | 0 | 0 | 0 |
| **learning** | 0 | 0 | 1 | 0 | 0 | 0 | 0 |
| **NLP** | 0 | 1 | 0 | 0 | 0 | 0 | 0 |
| **enjoy** | 1 | 0 | 0 | 0 | 0 | 0 | 1 |
| **flying** | 0 | 0 | 0 | 0 | 0 | 1 | 0 |

**Reading the matrix:**

- `I` occurs with `like` in **two** sentences → count = 2.
- `I` occurs with `enjoy` in **one** sentence → count = 1.
- `like` occurs with `deep` once and `NLP` once.

### The problem: sparsity and dimensionality

This matrix has serious issues:

| Issue | Why it matters |
|---|---|
| **Grows with vocabulary** | \(V \times V\) matrix. 100,000 words → 10 billion cells. |
| **High-dimensional** | Each word is a 100,000-dimensional vector. |
| **Very sparse** | Most words never co-occur. The matrix is mostly zeros. |
| **Storage cost** | Storing mostly zeros is wasteful. |
| **Sparsity issues in classifiers** | Downstream models become less robust when trained on sparse inputs. |

> **Analogy:** A co-occurrence matrix is like a giant attendance register with one row and column per student in a university. Most pairs never sit together, so the register is mostly blank. Huge, sparse, wasteful.

### The fix: dimensionality reduction

Store the important information in a **fixed, small number of dimensions** — a **dense vector**, usually **25 to 1,000 dimensions**.

**Methods:**

- **Singular Value Decomposition (SVD)** of the co-occurrence matrix:

\[
X = U \Sigma V^T
\]

where \(U\) and \(V\) are orthonormal matrices and \(\Sigma\) contains singular values.

- You can truncate to the top \(d\) singular values to get \(d\)-dimensional word vectors.

**Important caveat:** Dimensionality reduction doesn't always work perfectly. If it did, we wouldn't need Word2Vec or GloVe. But it's a useful starting point, and it motivated a better approach: **GloVe**.

---

## 5. GloVe: Meaning Lives in Ratios

### The origin

GloVe (**Global Vectors**) was introduced by **Pennington et al. in 2014**. It gave more meaningful representations for words with similar meanings, and experiments showed it was particularly helpful for **sentiment analysis** on emotional corpora compared to Word2Vec.

### The central insight

> **Word meaning is encoded in the ratios of co-occurrence probabilities.**

This is the sentence to remember. Not just the counts, not just the probabilities — the **ratios**.

### Running example: ice and steam

How can a machine learn that `ice` is related to `solid` and `steam` is related to `gas` **without a dictionary**?

Consider four sentences:

> `ice is solid.`  
> `ice melts in water.`  
> `steam is a gas.`  
> `water turns into steam.`

Build co-occurrence counts and convert them to probabilities:

\[
P(w_k \mid w_i) = \frac{X_{ik}}{X_i}
\]

where:

- \(X_{ik}\) = how many times word \(k\) appears in the context of word \(i\)
- \(X_i\) = total count for word \(i\)

### The probability table

|  | \(k = \text{solid}\) | \(k = \text{gas}\) | \(k = \text{water}\) | \(k = \text{fashion}\) |
|---|---|---|---|---|
| \(P(k \mid \text{ice})\) | **8.9** | 0.02 | 0.5 | 0.001 |
| \(P(k \mid \text{steam})\) | 0.02 | **8.5** | 0.5 | 0.001 |
| **Ratio** \(P(k \mid \text{ice}) / P(k \mid \text{steam})\) | **445** | **0.002** | **1.0** | **1.0** |

### Reading the ratios

| Ratio value | Meaning | Example |
|---|---|---|
| **≫ 1** | \(k\) is much more associated with `ice` than `steam` | `solid` |
| **≪ 1** | \(k\) is much more associated with `steam` than `ice` | `gas` |
| **≈ 1** | \(k\) does not help distinguish them | `water`, `fashion` |

> **Analogy:** `solid` is to `ice` as `gas` is to `steam`. The ratio says: "`solid` is a fingerprint for `ice`, not for `steam`." `water` appears with both, so its ratio is near 1 — it doesn't discriminate.

### The function \(f\)

GloVe asks: find a function of word vectors that captures this ratio:

\[
f(w_i, w_j, w_k) = \frac{P_{ik}}{P_{jk}}
\]

**Assumption 1:** \(f\) depends on the **vector difference** \(w_i - w_j\), because meaning relationships are differences in embedding space — the same spirit as Word2Vec vector arithmetic.

So we want:

\[
f(w_i - w_j, w_k) = \frac{P_{ik}}{P_{jk}}
\]

**Assumption 2:** The relationship between co-occurrence and word vectors should be **linear in the simplest possible way**.

**Assumption 3:** The model should be a **scalar-valued** single number measuring this ratio.

### Deriving the form

A natural choice that satisfies these assumptions is the exponential:

\[
f(w_i, w_j, w_k) = \exp\left((w_i - w_j) \cdot w_k\right)
\]

Why exponential? Because the left-hand side (a ratio of probabilities) must always be **positive**, and \(\exp(\cdot)\) guarantees positivity.

Set this equal to the ratio:

\[
\exp\left((w_i - w_j) \cdot w_k\right) = \frac{P_{ik}}{P_{jk}}
\]

Take the log of both sides:

\[
(w_i - w_j) \cdot w_k = \log P_{ik} - \log P_{jk}
\]

Rearrange:

\[
w_i \cdot w_k - \log P_{ik} = w_j \cdot w_k - \log P_{jk}
\]

The left side depends only on \(i\) and \(k\); the right side only on \(j\) and \(k\). For this to hold for all pairs, each side must be a function of \(k\) alone. GloVe absorbs these into **bias terms** \(b_i\) and \(b_k\):

\[
w_i \cdot w_k + b_i + b_k = \log X_{ik}
\]

Or, as an approximation:

\[
w_i \cdot w_k + b_i + b_k \approx \log X_{ik}
\]

> **What we're saying:** The dot product of the center word and context word vectors, plus biases, should approximate the **log of their co-occurrence count**.

### Why this matters

GloVe trains word vectors so that their dot products match **log co-occurrence counts**. It doesn't use a sliding window at every step; it uses the **global co-occurrence statistics** directly. This is why it's called **Global Vectors**.

---

## 6. The GloVe Loss Function

### The objective

\[
J = \sum_{i,j=1}^{V} f(X_{ij}) \left( w_i \cdot w_j + b_i + b_j - \log X_{ij} \right)^2
\]

Where:

- \(w_i \cdot w_j + b_i + b_j\) is the **prediction**
- \(\log X_{ij}\) is the **target**
- \(f(X_{ij})\) is a **weighting function**

### The weighting function

\[
f(X_{ij}) =
\begin{cases}
(X_{ij} / x_{\max})^\alpha & \text{if } X_{ij} < x_{\max} \\
1 & \text{otherwise}
\end{cases}
\]

With typical values \(x_{\max} = 100\) and \(\alpha = 3/4\).

**Properties:**

| Property | Why it matters |
|---|---|
| **Zero when \(X = 0\)** | Ignores non-occurrences. No need to compute loss for pairs that never appear. |
| **Grows for small counts** | Gives some weight to rare but real co-occurrences. |
| **Saturates for large counts** | Prevents very frequent pairs (like stop-word pairs) from dominating the loss. |

> **Key insight:** Very rare pairs are **noisy** — they might be accidental. Very frequent pairs **shouldn't dominate** — otherwise the model learns mostly about stop words. The weighting function balances these.

### Two vectors per word

GloVe gives each word **two vectors**:

- \(w_i\) — the **center** (word) vector
- \(u_i\) — the **context** vector

After training, you can:

- Use \(w_i\) directly
- Use \(u_i\) directly
- Use their sum or average \(w_i + u_i\) (often works best)

This is similar to Word2Vec, which also learns two vectors per word.

### GloVe vs. Word2Vec: a comparison

| Aspect | Word2Vec | GloVe |
|---|---|---|
| **Training signal** | Local context windows | Global co-occurrence counts |
| **What it captures** | Context similarity | Ratios of co-occurrence probabilities |
| **Computational cost** | Repeated window traversal | One pass to build matrix, then optimization |
| **Main advantage** | Good local representations | Captures global statistical structure |
| **Why it exists** | Efficient training | Meaning is in the ratios |

> **Important:** GloVe is **not primarily about reducing computational cost**. It's about the **semantic insight** that ratios of co-occurrence probabilities capture meaning in a way that simple context similarity does not.

---

## 7. Implementing Negative Sampling

### Step 1: Start from Word2Vec

Begin with a working Word2Vec implementation. The core training loop looks like this:

```python
for epoch in range(num_epochs):
    for center, context in pairs:
        # Compute positive loss
        pos_score = sigmoid(dot(E[center], W[context]))
        pos_loss = -log(pos_score)
        
        # Backpropagate and update
        ...
```

### Step 2: Add a function to sample negatives

```python
import random

def sample_negatives(vocab, k, exclude_word=None, distribution=None):
    """
    Sample k negative words from the vocabulary.
    
    Args:
        vocab: list of words in vocabulary
        k: number of negatives to sample
        exclude_word: word to avoid sampling (the true context)
        distribution: optional probability distribution for sampling
    
    Returns:
        list of k negative words
    """
    negatives = []
    while len(negatives) < k:
        if distribution:
            word = random.choices(vocab, weights=distribution, k=1)[0]
        else:
            word = random.choice(vocab)
        if word != exclude_word:
            negatives.append(word)
    return negatives
```

**Better: use the reshaped unigram distribution**

```python
def build_negative_distribution(corpus, power=0.75):
    """
    Build a sampling distribution using frequency^power.
    This dampens stop words and boosts rare words.
    """
    from collections import Counter
    counts = Counter(corpus)
    total = sum(counts.values())
    words = list(counts.keys())
    probs = [(counts[w] / total) ** power for w in words]
    # Normalize
    prob_sum = sum(probs)
    probs = [p / prob_sum for p in probs]
    return words, probs
```

### Step 3: Add the negative loss to the training loop

```python
for epoch in range(num_epochs):
    for center, context in pairs:
        # Positive pair
        pos_score = sigmoid(dot(E[center], W[context]))
        pos_loss = -log(pos_score)
        
        # Sample K negatives
        negatives = sample_negatives(vocab, K, exclude_word=context, 
                                      distribution=neg_probs)
        
        # Negative pairs
        neg_loss = 0
        for neg_word in negatives:
            neg_score = sigmoid(dot(E[center], W[neg_word]))
            neg_loss += -log(sigmoid(-dot(E[center], W[neg_word])))
        
        # Total loss
        total_loss = pos_loss + neg_loss
        
        # Backpropagate and update E and W
        ...
```

> **Note:** Only the loss changes. The update rules for `E` (center word vectors) and `W` (context word vectors) remain the same. You're just adding more terms to the gradient.

### Step 4: Plot embeddings with and without negative sampling

After training, reduce the embeddings to 2D (using PCA, t-SNE, or UMAP) and plot them:

```python
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def plot_embeddings(embeddings, words, title="Word Embeddings"):
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(embeddings)
    
    plt.figure(figsize=(10, 8))
    plt.scatter(reduced[:, 0], reduced[:, 1], alpha=0.5)
    for i, word in enumerate(words):
        plt.annotate(word, (reduced[i, 0], reduced[i, 1]))
    plt.title(title)
    plt.show()
```

**What to look for:**

- Do words with similar meanings cluster together?
- Do the clusters change when you add negative sampling?
- **Expected result:** The clusters should theoretically be **the same** with and without negative sampling, because negative sampling mainly improves **computational efficiency**, not the final semantic structure.

---

## 8. Implementing GloVe

### Step 1: Build the co-occurrence matrix

```python
from collections import defaultdict
import numpy as np

def build_cooccurrence_matrix(corpus, window_size=2):
    """
    Build a co-occurrence matrix from a corpus.
    
    Args:
        corpus: list of sentences (each sentence is a list of words)
        window_size: context window size on each side
    
    Returns:
        vocab: list of unique words
        cooc: dict mapping (i, j) to count
    """
    vocab = list(set(word for sentence in corpus for word in sentence))
    word_to_idx = {word: i for i, word in enumerate(vocab)}
    
    cooc = defaultdict(int)
    for sentence in corpus:
        for i, word in enumerate(sentence):
            center_idx = word_to_idx[word]
            for j in range(max(0, i - window_size), min(len(sentence), i + window_size + 1)):
                if i != j:
                    context_idx = word_to_idx[sentence[j]]
                    cooc[(center_idx, context_idx)] += 1
    
    return vocab, word_to_idx, cooc
```

### Step 2: Define the GloVe loss

```python
def glove_loss(w_i, w_j, b_i, b_j, X_ij, x_max=100, alpha=0.75):
    """
    Compute GloVe loss for a single pair.
    
    J = f(X_ij) * (w_i . w_j + b_i + b_j - log(X_ij))^2
    """
    if X_ij == 0:
        return 0  # Ignore non-occurrences
    
    weight = (X_ij / x_max) ** alpha if X_ij < x_max else 1.0
    prediction = np.dot(w_i, w_j) + b_i + b_j
    target = np.log(X_ij)
    
    return weight * (prediction - target) ** 2
```

### Step 3: Train with gradient descent

```python
def train_glove(cooc, vocab_size, embedding_dim=50, epochs=100, lr=0.05):
    """
    Train GloVe embeddings using gradient descent.
    """
    # Initialize parameters
    W = np.random.randn(vocab_size, embedding_dim) * 0.01
    U = np.random.randn(vocab_size, embedding_dim) * 0.01
    b = np.zeros(vocab_size)
    c = np.zeros(vocab_size)
    
    for epoch in range(epochs):
        total_loss = 0
        for (i, j), X_ij in cooc.items():
            if X_ij == 0:
                continue
            
            # Weight
            weight = (X_ij / 100) ** 0.75 if X_ij < 100 else 1.0
            
            # Prediction
            pred = np.dot(W[i], U[j]) + b[i] + c[j]
            target = np.log(X_ij)
            error = pred - target
            
            # Gradient
            grad = 2 * weight * error
            
            # Update (simplified SGD)
            W[i] -= lr * grad * U[j]
            U[j] -= lr * grad * W[i]
            b[i] -= lr * grad
            c[j] -= lr * grad
            
            total_loss += weight * error ** 2
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {total_loss:.4f}")
    
    # Combine center and context vectors
    embeddings = W + U
    return embeddings
```

### Step 4: Extract and use the vectors

```python
# After training
embeddings = train_glove(cooc, vocab_size=len(vocab), embedding_dim=50)

# Get vector for a word
word = "ice"
idx = word_to_idx[word]
vector = embeddings[idx]

# Find nearest neighbors
from sklearn.metrics.pairwise import cosine_similarity
similarities = cosine_similarity([vector], embeddings)[0]
nearest = np.argsort(similarities)[::-1][1:6]  # Exclude self
print([vocab[i] for i in nearest])
```

---

## 9. What to Expect When You Plot Embeddings

### With vs. without negative sampling

| Observation | Explanation |
|---|---|
| Loss curve is less smooth with negative sampling | Expected — you're adding stochastic negative samples, so the loss fluctuates more. |
| Clusters look similar | Expected — negative sampling mainly improves efficiency, not the final semantic structure. |
| Clusters look different | Possible with small corpora or insufficient training. **More data and more epochs** will help. |
| Some words move together | Normal — the model is still learning. With enough training, meaningful clusters emerge. |

### A practical note

With a tiny corpus, the plots may not make much sense. Word embeddings need **large corpora** (millions or billions of tokens) to produce meaningful clusters. For learning purposes, focus on the **code and the loss curves**; the 2D plots are a sanity check, not a definitive evaluation.

---

## 10. Preprocessing: Clean Data Beats Clever Models

Before training any embedding, preprocess your text:

| Step | Why |
|---|---|
| **Tokenization** | Split text into words/sentences. |
| **Lowercasing** | Reduce vocabulary size (optional). |
| **Stop-word removal** | Prevent the model from learning mostly about `the`, `of`, `a`. |
| **Lemmatization** | Reduce words to base form (`running` → `run`). |
| **POS tagging** | Useful for downstream tasks, sometimes for filtering. |

> **Key insight:** If you leave stop words in, the model will learn about stop words. Clean data produces clean embeddings. Text models work better with preprocessing.

---

## 11. Edge Cases and Practical Warnings

### The unseen pair problem

**Scenario:** In your corpus, you see `apple` with `ice` but never `apple` with `steam`. What happens to the ratio \(P(\text{apple} \mid \text{ice}) / P(\text{apple} \mid \text{steam})\)?

**Answer:** The ratio is undefined (division by zero). This is an edge case.

**Solutions:**

- **Smoothing:** Add a small constant to all counts.
- **Ignore:** Skip pairs with zero counts (GloVe's weighting function already does this).
- **Improve the algorithm:** GloVe was the first to use co-occurrence ratios this way; it's not the final word. You can design better methods.

### The \(K\) confusion

- In **negative sampling**, \(K\) = number of negative samples per positive pair. You choose it (e.g., 5–20).
- In **GloVe**, \(k\) is a context word index in \(X_{ik}\). It's not the same \(K\).

Don't confuse them.

### GloVe's scope

GloVe uses **global co-occurrence statistics**, not local windows. You can still choose a window size when building the co-occurrence matrix, but the core insight is about **ratios of global counts**, not local context.

### Adapting to other modalities

The co-occurrence idea can be adapted to:

- **Images:** Co-occurrence of visual features or objects.
- **Voice:** Voice is a discrete sequence (like text), so sequence models apply directly.
- **Video:** Continuous + temporal; may need architectural changes.

CNNs were originally for images but have been adapted for text. Similarly, co-occurrence ideas can transfer across modalities with appropriate modifications.

---

## 12. Where This Goes Next

After mastering word embeddings, the natural next steps are:

1. **Evaluation:** How do you know if your embeddings are good?
   - **Intrinsic evaluation:** Word similarity, analogy tasks.
   - **Extrinsic evaluation:** Performance on downstream tasks (sentiment analysis, NER, etc.).

2. **RNN and Transformer architectures:** Word embeddings are the input layer. To build conversational AI, translation systems, or coreference resolution, you need sequence models.

3. **Coreference resolution:** Deciding whether "he" refers to "John" or "Bill" requires understanding context — embeddings are the foundation.

4. **Conversational systems:** Training these systems requires careful **data formatting** and adapting architectures to the task. The embedding is just the first step.

> **Important:** The way you train these systems requires a lot of formatting of how you put data as input. How you adapt your RNN or Transformer to different AI applications is completely different from the embedding step.

---

## 13. One-Page Cheat Sheet

| Concept | Key Idea | Memory Hook |
|---|---|---|
| **Word2Vec cost** | Denominator sums over >100k vocab | "Check every shop item for every sale" |
| **Negative sampling** | Binary classification: real vs. noisy pairs | "Sample the few, not the all" |
| **Positive loss** | \(-\log \sigma(v_c \cdot u_w)\) | Pull `cat–sat` together |
| **Negative loss** | \(-\sum \log \sigma(-v_c \cdot u_{w'})\) | Push `cat–apple` apart |
| **Sampling negatives** | Raw unigram gives stop words | Reshape with \(U^{3/4}\) or similar |
| **Stop words** | Remove in preprocessing; else model learns them | Clean data → clean embeddings |
| **Co-occurrence matrix** | Count how often words appear near each other | Giant sparse attendance register |
| **Co-occurrence problems** | High-dimensional, sparse, storage-heavy | Mostly blank matrix |
| **Fix** | Dimensionality reduction (SVD), dense 25–1000 dims | Compress the register |
| **GloVe insight** | Ratios of co-occurrence probabilities encode meaning | `solid:ice :: gas:steam` |
| **GloVe target** | \(w_i \cdot w_j + b_i + b_j \approx \log X_{ij}\) | Predict log co-occurrence |
| **GloVe weight** | Zero at 0, grows small, saturates large | Ignore noise, cap frequent pairs |
| **GloVe vs. Word2Vec** | Ratios vs. context similarity; not mainly speed | "Ratios, not just similarity" |
| **Coding** | Add K-negative loop, negative loss, compare plots | "Sample → loss → update → plot" |
| **Clusters** | Theoretically same with/without negatives | Negatives mainly save compute |
| **Edge cases** | Unseen pairs need smoothing or handling | Zero counts break ratios |

---

## Summary: The Two Big Ideas in One Sentence Each

1. **Negative sampling:** Instead of comparing a center word with the entire vocabulary, train a classifier to distinguish real context pairs from a few noisy ones — pull true pairs together, push fake pairs apart.

2. **GloVe:** Word meaning is encoded in the ratios of co-occurrence probabilities; train vectors so their dot products match log co-occurrence counts, with a weighting function that ignores noise and caps frequent pairs.

> **The unifying theme:** Both methods replace expensive full-vocabulary computations with smarter objectives — one by sampling, the other by counting and factorizing. Both produce dense, meaningful word vectors that serve as the foundation for everything that follows in NLP.

---

*This tutorial is designed to be self-contained and open-source ready. All code examples are illustrative; adapt them to your specific framework (NumPy, PyTorch, TensorFlow) as needed.*