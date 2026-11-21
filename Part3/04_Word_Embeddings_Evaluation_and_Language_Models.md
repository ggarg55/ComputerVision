# Word Embeddings Evaluation & Language Models: A Complete Tutorial

> **What You'll Learn:** How to evaluate word embeddings, understand their limitations, and build your first language model from scratch using PyTorch.

---

## Table of Contents

1. [Introduction: Why Evaluation Matters](#1-introduction-why-evaluation-matters)
2. [Intrinsic vs. Extrinsic Evaluation](#2-intrinsic-vs-extrinsic-evaluation)
3. [Intrinsic Evaluation Methods](#3-intrinsic-evaluation-methods)
4. [Extrinsic Evaluation Methods](#4-extrinsic-evaluation-methods)
5. [Which Embedding Should You Choose?](#5-which-embedding-should-you-choose)
6. [Problems with Word2Vec and GloVe](#6-problems-with-word2vec-and-glove)
7. [Solutions: Subword Embeddings and FastText](#7-solutions-subword-embeddings-and-fasttext)
8. [Multilingual Embeddings](#8-multilingual-embeddings)
9. [Language Models: The Foundation](#9-language-models-the-foundation)
10. [Neural Language Models](#10-neural-language-models)
11. [Building an N-Gram Language Model with PyTorch](#11-building-an-n-gram-language-model-with-pytorch)
12. [Limitations and the Path Forward](#12-limitations-and-the-path-forward)

---

## 1. Introduction: Why Evaluation Matters

You've trained a Word2Vec model. You have 300-dimensional vectors for every word in your vocabulary. Now what?

**The critical question:** How do you know these embeddings are actually meaningful? How do you know if all that training accomplished something useful?

This tutorial answers that question and takes you on a journey from evaluating word embeddings to building your first language model. By the end, you'll understand:

- How to measure embedding quality (both intrinsically and extrinsically)
- The fundamental problems with popular embeddings like Word2Vec and GloVe
- How FastText and subword embeddings solve these problems
- How to build an N-gram language model using PyTorch

---

## 2. Intrinsic vs. Extrinsic Evaluation

There are two fundamentally different approaches to evaluating word embeddings:

```
Word Embeddings
      │
      ├── Intrinsic Evaluation (Test on subtasks)
      │        ├── Fast to compute
      │        ├── Helps understand the system
      │        └── Disadvantage: May not correlate to real tasks
      │
      └── Extrinsic Evaluation (Test on real tasks)
               ├── More meaningful
               ├── Time-consuming
               └── Directly measures usefulness
```

### Intrinsic Evaluation

Intrinsic evaluation tests embeddings on **specific intermediate subtasks** — not real-world tasks. These methods are:

- **Fast to compute** — You can run them in seconds
- **Helpful for understanding** — They tell you what your embedding has learned
- **Limited** — Success on a subtask doesn't guarantee success on real tasks

**Analogy:** Think of intrinsic evaluation like practicing in the nets. You can quickly test your batting technique, but it doesn't tell you if you can perform in an actual match.

### Extrinsic Evaluation

Extrinsic evaluation tests embeddings on **real tasks** like:

- Machine translation
- Sentiment analysis
- Question answering
- Named entity recognition

These methods are:

- **More meaningful** — They measure what actually matters
- **Time-consuming** — You need to train a full model on a downstream task
- **Directly useful** — Success here means your embeddings are valuable

**Analogy:** Extrinsic evaluation is playing an actual match. It takes longer, but tells you if you can perform when it matters.

---

## 3. Intrinsic Evaluation Methods

### 3.1 Word Similarity (Relatedness)

**The concept:** Compute cosine similarity between word vectors and compare it to human judgment.

**Example:**

| Word Pair | Human Similarity | Expected Cosine Similarity |
|-----------|------------------|---------------------------|
| car – automobile | 95% | High (e.g., 0.92) |
| car – banana | 5% | Low (e.g., 0.12) |
| happy – joyful | 90% | High (e.g., 0.89) |

If your model gives `car–automobile` a low similarity score but `car–banana` a high score, your embeddings are broken!

**Common datasets:**
- **WordSim-353** — 353 word pairs with human similarity ratings
- **SimLex-999** — 999 word pairs focused on similarity (not relatedness)
- **MEN** — 3,000 word pairs

**How to compute:**

```python
from sklearn.metrics.pairwise import cosine_similarity

def word_similarity(word1, word2, embeddings):
    vec1 = embeddings[word1].reshape(1, -1)
    vec2 = embeddings[word2].reshape(1, -1)
    return cosine_similarity(vec1, vec2)[0][0]

# Example
similarity = word_similarity("car", "automobile", embeddings)
print(f"car–automobile similarity: {similarity:.3f}")
```

### 3.2 Analogy Task

**The classic example:**

```
man : king :: woman : ?
```

The answer should be **queen**. This works because word embeddings capture semantic relationships as vector arithmetic:

```
vector(king) - vector(man) + vector(woman) ≈ vector(queen)
```

**How it works mathematically:**

```
king - man + woman = ?

Step 1: king - man = "royalty" direction
Step 2: "royalty" + woman = queen
```

**Visual representation:**

```
        king
         ▲
        / \
       /   \
      /     \
   man ───── woman
      \     /
       \   /
        \ /
         ▼
       queen
```

**Common datasets:**
- **Google Analogy Test Set** — ~20,000 analogy questions
- **BATS** — More challenging analogies across 40 categories

**Example questions from Google's dataset:**

| Relationship | Example 1 | Example 2 |
|--------------|-----------|-----------|
| Country–Capital | Paris : France :: Tokyo : ? | Rome : Italy :: ? |
| Currency | dollar : USA :: yen : ? | rupee : India :: ? |
| Gender | king : queen :: man : ? | father : mother :: ? |
| Tense | walking : walked :: swimming : ? | run : ran :: ? |

**Implementation:**

```python
def analogy(word1, word2, word3, embeddings):
    """
    Solve: word1 : word2 :: word3 : ?
    Returns the most similar word to (word2 - word1 + word3)
    """
    vec = embeddings[word2] - embeddings[word1] + embeddings[word3]
    
    # Find closest word (excluding input words)
    similarities = cosine_similarity(vec.reshape(1, -1), embeddings)[0]
    best_idx = np.argsort(similarities)[::-1]
    
    for idx in best_idx:
        candidate = idx_to_word[idx]
        if candidate not in [word1, word2, word3]:
            return candidate
```

### 3.3 Categorization / Clustering

**The concept:** If the embedding groups words of the same category together, it's a good embedding.

**Visual example:**

```
Cluster 1 (Animals):    dog, cat, horse, elephant, tiger
Cluster 2 (Fruits):     apple, orange, banana, mango, grape
Cluster 3 (Vehicles):   car, truck, bicycle, motorcycle, bus
Cluster 4 (Colors):     red, blue, green, yellow, purple
```

**Quality metrics:**

- **Cluster purity** — What fraction of words in a cluster belong to the same category?
- **F1 score** — Harmonic mean of precision and recall for clustering

**Example:**

If a cluster labeled "Animals" contains 8 animals and 2 fruits:
```
Purity = 8/10 = 80%
```

**Why this matters:** Good embeddings should naturally group semantically similar words together without any supervision.

### 3.4 Selectional Preference

**The concept:** Measure how well embeddings capture typical syntactic and semantic relations — like verb-object pairs.

**Examples:**

| Verb | Object | Should Score |
|------|--------|--------------|
| eat | apple | HIGH |
| eat | rock | LOW |
| drink | water | HIGH |
| drink | rock | LOW |
| read | book | HIGH |
| read | rock | LOW |

**Why this is relevant today:** This technique is used to interpret large language models. Researchers extract embeddings from every layer and test them on small intermediate tasks to understand what each layer captures.

```
Layer 1  → Tests POS tagging → Poor performance → Layer 1 doesn't capture syntax
Layer 2  → Tests POS tagging → Good performance → Layer 2 captures syntax
Layer 15 → Tests verb-noun relations → Good performance → Layer 15 captures semantics
```

**Modern relevance:** This is exactly how researchers probe large language models today! They ask: Does layer 2 understand syntax? Does layer 15 understand semantics? This technique started with word embeddings!

---

## 4. Extrinsic Evaluation Methods

### 4.1 The Core Concept

Extrinsic evaluation tests embeddings on **actual tasks** like:

- Machine translation
- Sentiment analysis
- Question answering
- Named entity recognition
- POS tagging

**Key insight:** A downstream task is a well-defined NLP problem — not just similarity finding.

### 4.2 Fixed vs. Fine-tuned Embeddings

When using embeddings for a downstream task, you have two options:

| Aspect | Fixed (Pre-trained) | Fine-tuned |
|--------|---------------------|------------|
| **What happens?** | Vectors stay frozen | Vectors get updated during training |
| **Pros** | Fewer parameters, less overfitting | Task-adaptive |
| **Cons** | May not adapt to task-specific meaning | Risk of forgetting general knowledge |

**Example: The word "bank"**

- **General meaning:** Financial institution
- **River context:** Side of a river

With fixed embeddings, the model only knows the general meaning. Fine-tuning helps adapt to task-specific meanings.

**Implementation comparison:**

```python
# Fixed embeddings
embedding_layer = nn.Embedding.from_pretrained(pretrained_weights, freeze=True)

# Fine-tuned embeddings
embedding_layer = nn.Embedding.from_pretrained(pretrained_weights, freeze=False)
```

### 4.3 The Combined Approach

You can also combine both: pre-trained embeddings + learned task-specific embeddings.

**Advantages:**
- More expressive
- Merges general + task-specific info

**Disadvantages:**
- Increases model size and computation cost

**Implementation:**

```python
class CombinedEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_dim, pretrained_weights):
        super().__init__()
        # Fixed pre-trained embeddings
        self.pretrained = nn.Embedding.from_pretrained(pretrained_weights, freeze=True)
        # Learnable task-specific embeddings
        self.task_specific = nn.Embedding(vocab_size, embed_dim)
        
    def forward(self, x):
        return torch.cat([self.pretrained(x), self.task_specific(x)], dim=-1)
```

### 4.4 Evaluation Metrics

**For Machine Translation: BLEU Score**

BLEU (Bilingual Evaluation Understudy) compares n-grams between your translation and reference translations.

```
BLEU = BP × exp(Σ w_n × log(p_n))

Where:
- p_n = precision of n-grams
- BP = brevity penalty (penalizes short translations)
- w_n = weights (typically 1/4 each for 1-4 grams)
```

**Example:**

```
Reference: "The cat sat on the mat"
Candidate: "The cat is on the mat"

1-gram matches: The, cat, on, the, mat = 5/6 = 0.833
2-gram matches: The cat, on the, the mat = 3/5 = 0.600
3-gram matches: on the mat = 1/4 = 0.250
4-gram matches: 0/3 = 0

BLEU ≈ 0.45
```

**For Classification: F1 Score**

```
F1 = 2 × (Precision × Recall) / (Precision + Recall)

Precision = True Positives / (True Positives + False Positives)
Recall = True Positives / (True Positives + False Negatives)
```

**Example:**

```
Sentiment Analysis Results:
- True Positives (correctly predicted positive): 80
- False Positives (incorrectly predicted positive): 20
- False Negatives (missed positive): 10

Precision = 80 / (80 + 20) = 0.80
Recall = 80 / (80 + 10) = 0.89
F1 = 2 × (0.80 × 0.89) / (0.80 + 0.89) = 0.84
```

---

## 5. Which Embedding Should You Choose?

### The Answer: No One Size Fits All

Different embeddings perform differently on different tasks. There is no single best embedding.

**Comparison of popular embeddings:**

| Embedding | Relatedness | Categorization | Analogy |
|-----------|-------------|----------------|---------|
| SVD | Good | Varies | Varies |
| GloVe | Good | Good | Good |
| Word2Vec | Good | Good | Good |

**Recommendation:** It's completely experimental. Try different embeddings, evaluate on your task, and use what works best. Hit and try!

---

## 6. Problems with Word2Vec and GloVe

### 6.1 Problem 1: Superficial Differences

**The issue:** These embeddings capture completely different vectors for "dog" and "dogs" even though they should be similar.

```
"dog"  → [0.2, 0.5, -0.1, ...]  (300-dim vector)
"dogs" → [0.8, -0.3, 0.4, ...]  (completely different!)
```

**Why this happens:** These models operate at the word level. They ignore subword or morphological structure.

**Impact:**

- Model doesn't understand that "dogs" = "dog" + "s" (plural)
- Rare words get poor embeddings
- Morphological variants like "walk" and "walking" get different embeddings

### 6.2 Problem 2: Not Coordinated with Factual Knowledge

**The issue:** Embeddings capture statistical patterns in text, not factual or logical knowledge.

**Example: Whale and Fish**

```
Vector Space:

    fish *         * whale
         \       /
          \     /
           \   /
            \ /
         * shark
```

In the embedding space, whale is close to fish and shark because they appear in similar contexts (ocean, swimming, etc.). But biologically, whale is a mammal, not a fish. The embedding captures **statistical co-occurrence**, not **factual truth**.

**Another example:** Dog and perro (dog in Spanish) might not be aligned together. That's why multilingual embeddings are needed.

### 6.3 Problem 3: Not Interpretable

**The issue:** We make a 300-dimension vector for a word. We don't know what these values represent.

```
Vector for "king": [0.23, -0.45, 0.67, 0.12, ...] (300 numbers)

Questions we can't answer:
- What does 0.23 represent?
- What does -0.45 represent?
- What does dimension 5 mean?
```

**Recent research:** Some methods have tried interpreting these dimensions, but they remain largely uninterpretable.

### 6.4 Problem 4: Encoding Bias

**The issue:** Embeddings capture biases present in the training data.

**Classic example:**

```
man : computer programmer :: woman : ?
```

The model would predict **homemaker**. The word next to "women" was coming as more as "homemaker."

**More examples of bias:**

| Input | Biased Output |
|-------|---------------|
| doctor - he + she | nurse |
| engineer - he + she | secretary |
| father - man + woman | mother |
| CEO - man + woman | secretary |

**Why this happens:** The model learns from real-world data, which contains human biases. If we understand the cognitive process, we might be able to tell the model to not do that.

**Solutions:**

- Post-processing/debiasing techniques
- Balanced or filtered corpora
- Manual filtering of biased samples

**Honest truth:** You cannot remove bias until humans are unbiased. This is an active area of research.

---

## 7. Solutions: Subword Embeddings and FastText

### 7.1 Character-Based Embeddings

**The idea:** Instead of treating words as atomic units, represent them as sequences of characters.

```
"cat" → [embed("c"), embed("a"), embed("t")]
           ↓          ↓          ↓
       [0.1,...]  [0.5,...]  [0.3,...]
```

**How to compose:** Pass these character-level embeddings to a sequence model (like an RNN) to get the word representation.

### 7.2 FastText: Character N-Grams

**The idea:** Represent each word as a bag of character n-grams.

**Example: "playing" with n=3**

```
Character n-grams:
pla, lay, ayi, yin, ing

Plus the whole word:
<playing>
```

**How FastText creates word vectors:**

```
vector("playing") = average(
    vector("pla"),
    vector("lay"),
    vector("ayi"),
    vector("yin"),
    vector("ing"),
    vector("<playing>")
)
```

**Why this solves the problem:**

```
"dog" = embed("d") + embed("o") + embed("g") + embed("<dog>")
"dogs" = embed("d") + embed("o") + embed("g") + embed("s") + embed("<dogs>")
```

They share the n-grams "d", "o", "g" — so their embeddings will be similar!

**Implementation:**

```python
def get_ngrams(word, n=3):
    """Extract character n-grams from a word."""
    ngrams = []
    # Add boundary markers
    padded = f"<{word}>"
    for i in range(len(padded) - n + 1):
        ngrams.append(padded[i:i+n])
    return ngrams

# Example
print(get_ngrams("playing"))
# ['<pl', 'pla', 'lay', 'ayi', 'yin', 'ing', 'ng>']
```

### 7.3 Morpheme-Based Embeddings

**The idea:** Break words into morphemes (smallest meaningful units).

**Example: "unfortunately"**

```
unfortunately = un- + fortunate + -ly
                (negation) (root) (adverb marker)
```

**Why this matters:** Each morpheme should have a unique embedding. Words sharing the root word "fortunate" should have similar embeddings.

**Challenges:**

- Need to capture morphological regularities
- Different languages have different morphological rules

### 7.4 Solving the Out-of-Vocabulary (OOV) Problem

**The problem:** What happens when you encounter a word not seen during training?

**FastText solution:** Compose the embedding from character n-grams.

**Example: "unknown"**

```
If "unknown" is not in vocabulary but "un" is:

vector("unknown") = average(
    vector("unk"),
    vector("nkn"),
    vector("kno"),
    vector("now"),
    vector("own"),
    ...
)
```

**Alternative approaches:**

1. **Simple pooling:** Average the embeddings of known n-grams
2. **Pass through LSTM:** Learn to compose character embeddings

---

## 8. Multilingual Embeddings

### 8.1 The Concept of Embedding Spaces

Different languages learn different embedding spaces. The goal is to align them.

```
English Space:                  Chinese Space:
    dog                            ma (dog)
    cat                            mao (cat)
    house                          fangzi (house)

Goal: Align these spaces so that:
    dog ↔ ma (should be close)
    cat ↔ mao (should be close)
```

### 8.2 Matrix Transformation Approach

**The idea:** Find a transformation matrix W such that:

```
English_embedding × W ≈ Chinese_embedding
```

**Key insight:** No need to build a dictionary from scratch. Just work with the representations you've learned separately and bring them closer through transformation.

**Implementation:**

```python
def align_embeddings(english_embeds, chinese_embeds, parallel_dict):
    """
    Learn a linear transformation to align two embedding spaces.
    
    parallel_dict: Dictionary mapping English words to Chinese words
    """
    # Create matrices from parallel dictionary
    X = np.array([english_embeds[w] for w in parallel_dict.keys()])
    Y = np.array([chinese_embeds[parallel_dict[w]] for w in parallel_dict.keys()])
    
    # Solve for W: XW ≈ Y (using least squares)
    W, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
    
    return W

# Apply transformation
aligned_english = english_embeds @ W
```

**Why linear algebra is important:** Ultimately, people are going to think in the latent space. They're not going to think discretely. This is why understanding linear algebra is crucial for NLP.

---

## 9. Language Models: The Foundation

### 9.1 What is a Language Model?

**Simple definition:** A language model predicts or generates text.

**The fundamental question:** How to learn a language model?

Before deep learning, we used **N-gram language models**.

### 9.2 The N-Gram Language Model

**The concept:** Collect statistics about how frequent different n-grams are and use this to predict the next word.

**The Markov Assumption:**

```
P(w_t | w_1, w_2, ..., w_{t-1}) ≈ P(w_t | w_{t-n+1}, ..., w_{t-1})
```

The prediction of the next word depends only on the previous n-1 words.

### 9.3 A Simple Example

**The full sentence:** "As the proctor started the clock, the students opened their ___"

**For a 4-gram model:**

We discard everything before the last three words:
```
Context: "students opened their"
Question: What comes next?
```

**The counting method:**

```
P(w | "students opened their") = 
    Count("students opened their" + w) / Count("students opened their")
```

**Example calculation:**

If "students opened their" occurred 1,000 times:

| Next Word | Count | Probability |
|-----------|-------|-------------|
| books | 400 | 0.40 |
| exams | 100 | 0.10 |
| laptops | 50 | 0.05 |

**Prediction:** "books" (highest probability!)

**The limitation:** Should we have considered the context? "Students opened their books" should NOT have been predicted because they are already sitting in an exam. If it had considered the word "proctor" into context, it would not have predicted books.

### 9.4 Problems with N-Gram Models

1. **Sparsity Problem:** What if "students opened their books" never occurred in the data? The probability becomes zero.

2. **Infinite Denominator Problem:** What if "students opened their" never appeared in the data? It goes infinite.

3. **Storage Problem:** As the corpus grows, the number of n-grams grows exponentially.

### 9.5 Generating Text with N-Grams

**Example from financial news:**

```
Given: "today the"
Predict: "price" (highest probability)
Update: "today the price"
Predict: "of"
Update: "today the price of"
Predict: "gold"
...

Result: "today the price of gold per turn while production of shoe last"
```

**Observation:** It's incoherent but somewhat grammatical. Some lines are probably correctly representing grammar. The more data you give, the better the model becomes.

---

## 10. Neural Language Models

### 10.1 The Idea

Instead of counting probabilities, use neural networks to predict the next word.

```
Input: Word embeddings for context words
       ↓
Hidden Layer (fully connected)
       ↓
Output: Softmax over vocabulary
       ↓
Prediction: Probability distribution over next word
```

### 10.2 A Concrete Example

**Vocabulary:** `["books", "houses", "lamps", "stamps"]`

**Input:** Context word embeddings (3-dimensional for simplicity)

```
x = [2.3, 0.9, 5.4]  (embedding of context word)
```

**Weight matrix:** `W` has shape `(4, 3)` — 4 vocabulary words, 3 dimensions

```
W = [[1.2, -0.3, 0.9],
     [0.5, 0.7, -0.2],
     [-0.1, 0.4, 0.8],
     [0.3, -0.6, 0.1]]
```

**Forward pass:**

```
Wx = [1.2×2.3 + (-0.3)×0.9 + 0.9×5.4,  ...]
   = [7.35, 0.65, 3.94, 0.27]
```

**Apply softmax:**

```
softmax([7.35, 0.65, 3.94, 0.27]) = [0.24, 0.006, 0.73, 0.02]
```

**Interpretation:**

| Word | Probability |
|------|-------------|
| books | 0.24 |
| houses | 0.006 |
| lamps | 0.73 |
| stamps | 0.02 |

**Prediction:** "lamps" (highest probability)

**Training:** If the ground truth is "books", compute error between "lamps" and "books", backpropagate, and update weights.

### 10.3 Adapting to Different Tasks

The same architecture can be adapted to any task:

| Task | Input | Output |
|------|-------|--------|
| Language Modeling | Previous words | Next word |
| Machine Translation | Sentence in language A | Sentence in language B |
| Sentiment Analysis | Sentence | Positive/Negative label |
| POS Tagging | Word | Part of speech |

**Key insight:** It's only a matter of fact how you pass the input and what you want to optimize at the end.

---

## 11. Building an N-Gram Language Model with PyTorch

### 11.1 Overview

We'll build an N-gram language model using PyTorch. The steps are:

1. Prepare your data
2. Create vocabulary
3. Build the model
4. Train the model
5. Generate text

### 11.2 Step 1: Prepare Your Data

Use an existing dataset from PyTorch's library:

```python
import torch
from torch.utils.data import Dataset, DataLoader
from torchtext.datasets import IMDB

# Load IMDB dataset
train_iter, test_iter = IMDB(split=('train', 'test'))
```

### 11.3 Step 2: Create Vocabulary

```python
from collections import Counter
import re

def tokenize(text):
    """Simple tokenizer."""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text.split()

def build_vocab(texts, min_freq=2):
    """Build vocabulary from texts."""
    counter = Counter()
    for text in texts:
        counter.update(tokenize(text))
    
    # Add special tokens
    vocab = {'<pad>': 0, '<unk>': 1}
    
    for word, freq in counter.items():
        if freq >= min_freq:
            vocab[word] = len(vocab)
    
    return vocab
```

### 11.4 Step 3: Create Dataset

```python
class NgramDataset(Dataset):
    def __init__(self, texts, vocab, n=3):
        self.n = n
        self.vocab = vocab
        self.data = []
        
        for text in texts:
            tokens = tokenize(text)
            token_ids = [vocab.get(t, vocab['<unk>']) for t in tokens]
            
            # Create n-gram pairs
            for i in range(len(token_ids) - n):
                context = token_ids[i:i+n-1]
                target = token_ids[i+n-1]
                self.data.append((context, target))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        context, target = self.data[idx]
        return torch.tensor(context), torch.tensor(target)
```

### 11.5 Step 4: Build the Model

```python
import torch.nn as nn

class NgramLanguageModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, context_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.linear = nn.Linear(embed_dim * context_size, vocab_size)
    
    def forward(self, context):
        # context: (batch_size, context_size)
        embeds = self.embedding(context)  # (batch_size, context_size, embed_dim)
        embeds = embeds.view(embeds.size(0), -1)  # (batch_size, context_size * embed_dim)
        logits = self.linear(embeds)  # (batch_size, vocab_size)
        return logits
```

### 11.6 Step 5: Training Loop

```python
def train(model, dataloader, epochs=10, lr=0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        total_loss = 0
        for context, target in dataloader:
            optimizer.zero_grad()
            logits = model(context)
            loss = criterion(logits, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(dataloader):.4f}")
```

### 11.7 Step 6: Generate Text

```python
def generate(model, start_context, vocab, inv_vocab, num_words=50):
    """Generate text given a starting context."""
    model.eval()
    context = [vocab.get(w, vocab['<unk>']) for w in start_context.split()]
    result = start_context.split()
    
    with torch.no_grad():
        for _ in range(num_words):
            context_tensor = torch.tensor([context[-model.context_size:]])
            logits = model(context_tensor)
            probs = torch.softmax(logits, dim=-1)
            
            # Sample from distribution
            next_word_id = torch.multinomial(probs, 1).item()
            next_word = inv_vocab[next_word_id]
            
            result.append(next_word)
            context.append(next_word_id)
    
    return ' '.join(result)

# Example
start = "the student opened"
generated = generate(model, start, vocab, inv_vocab, num_words=30)
print(generated)
```

### 11.8 Expected Output

**With a small dataset:**

```
"the student opened laptop laptop laptop laptop laptop..."
```

**The problem:** Repetition! This is common with simple language models.

**With a larger dataset:**

```
"the student opened their books and began to read the first chapter..."
```

**The solution:** More data and more context (larger n) leads to better generation.

---

## 12. Limitations and the Path Forward

### 12.1 The Repetition Problem

**What you'll observe:** Neural language models tend to repeat themselves.

```
"loud guitar pres is loud guitar loud..."
```

**Why this happens:** The model learns probability distributions from limited data. It doesn't understand the broader context.

**Historical context:** When there were no transformer models, this repetition was very common with RNNs and other neural networks.

### 12.2 The Context Limitation

**The issue:** N-gram models only look at n-1 previous words.

**Example:**

```
Context: "students opened their"
Prediction: "books"

But what if the context was:
"As the proctor started the clock, the students opened their ___"

The word "proctor" should influence the prediction!
(Maybe "exam sheets" instead of "books")
```

**N-gram models can't capture this** — they only see the last n-1 words.

### 12.3 The Scaling Solution

**The insight:** Transformers can attend to the entire context, not just the last n words.

**Comparison:**

| Model | Context Window | Performance |
|-------|----------------|-------------|
| N-gram (n=3) | 2 previous words | Poor fluency |
| RNN | Full sequence (but forgets) | Better, but repetition issues |
| Transformer | Full sequence (with attention) | Excellent fluency and coherence |

**Why transformers work:**

```
With transformers:
- Can attend to any part of the input
- Scales with more data
- Generalizes better
- Solves the repetition problem
```

### 12.4 The Journey Ahead

```
Word Embeddings → Evaluation → Problems → Solutions → Language Models → Transformers

1. Word2Vec/GloVe: Learn word vectors
2. Intrinsic Evaluation: Test on subtasks
3. Extrinsic Evaluation: Test on real tasks
4. Problems: Superficial differences, bias, non-interpretability
5. Solutions: FastText, character n-grams, multilingual alignment
6. Language Models: Predict next word
7. N-gram Models: Count probabilities
8. Neural Language Models: Learn representations
9. Limitations: Context window, repetition, scalability
10. Solution: Transformers (next chapter!)
```

---

## Summary Table: All Concepts Covered

| Concept | Key Idea | Example |
|---------|----------|---------|
| **Intrinsic Evaluation** | Test on subtasks | Relatedness: car-automobile should be 0.95 |
| **Extrinsic Evaluation** | Test on real tasks | Machine translation, sentiment analysis |
| **Word Similarity** | Compare cosine similarity to human scores | car-automobile: high, car-banana: low |
| **Analogy Task** | Vector arithmetic | king - man + woman = queen |
| **Categorization** | Cluster similar words | Animals: dog, cat, horse |
| **Selectional Preference** | Verb-noun compatibility | eat-apple: high, eat-rock: low |
| **Fixed Embeddings** | Don't update vectors | Pre-trained embeddings stay frozen |
| **Fine-tuned Embeddings** | Update vectors for task | Adapt "bank" for financial context |
| **F1 Score** | Precision + Recall combination | 2PR/(P+R) |
| **BLEU Score** | N-gram overlap metric | Compare translation to reference |
| **Superficial Differences** | dog vs dogs problem | FastText solves with n-grams |
| **Factual Knowledge** | Statistical ≠ factual | Whale close to fish (but wrong!) |
| **Interpretability** | Can't understand 300-dim vectors | What does dimension 23 mean? |
| **Bias** | Encodes stereotypes | programmer - he + she = homemaker |
| **FastText** | Character n-grams | playing = pla+lay+ayi+ying |
| **Morphological Embeddings** | Morphemes as units | unfortunately = un+fortunate+ly |
| **Multilingual Embeddings** | Align embedding spaces | English "cat" ↔ Chinese "mao" |
| **N-gram Language Model** | Count probabilities | "students opened" → predict next word |
| **Markov Assumption** | Limited context | Only last n words matter |
| **Neural Language Model** | NN for prediction | Embeddings → Hidden → Softmax |
| **Repetition Problem** | Models repeat themselves | "loud guitar pres is loud guitar" |
| **Sparsity Problem** | Zero probability for unseen | "students opened there books" never seen |
| **Scaling Solution** | Transformers + more data | GPT models generalize better |

---

## Key Takeaways

1. **Evaluation is crucial:** You cannot improve what you cannot measure.

2. **No one-size-fits-all:** Different embeddings work better for different tasks. Experiment!

3. **Understanding problems leads to solutions:** FastText came from understanding that word-level models ignore morphology.

4. **Simple can be powerful:** A simple n-gram language model can generate coherent text by counting probabilities.

5. **Limitations drive innovation:** The repetition problem in neural language models led to transformer architectures.

6. **Linear algebra matters:** Understanding transformations is crucial for multilingual embeddings and beyond.

7. **The journey continues:** What you've learned is the foundation for understanding transformers, BERT, GPT, and modern NLP.

---

## Further Reading

- **Word Embeddings:** Mikolov et al., "Efficient Estimation of Word Representations in Vector Space" (2013)
- **GloVe:** Pennington et al., "GloVe: Global Vectors for Word Representation" (2014)
- **FastText:** Bojanowski et al., "Enriching Word Vectors with Subword Information" (2017)
- **Multilingual Embeddings:** Conneau et al., "Word Translation Without Parallel Data" (2018)
- **Language Models:** Bengio et al., "A Neural Probabilistic Language Model" (2003)

---

*This tutorial is designed to be a standalone resource for learning about word embeddings and language models. It covers the same concepts as the original lecture but is structured for self-paced learning.*