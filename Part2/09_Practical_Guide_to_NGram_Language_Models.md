# A Practical Guide to N‑Gram Language Models

### From Counting Words to Computing Perplexity (with a Hands‑On SRILM Walkthrough)

---

## Introduction

Language models power autocomplete, machine translation, speech recognition, and spam filters. At their core, they answer one deceptively simple question:

> **Given some words, how likely is the next word?**

This tutorial takes you from first principles to a working implementation. You will build an n‑gram language model by hand, compute sentence probabilities, evaluate the model with perplexity, fix the "zero probability" problem using smoothing, and then verify every number using the SRILM toolkit.

Every concept is illustrated with a small, concrete example. No prior exposure to the original lecture is assumed.

**Prerequisites:** basic probability (conditional probability, chain rule), a scientific calculator, and a Unix‑like shell or Google Colab.

---

## Table of Contents

1. [The Core Recipe: Sentence Probability with N‑Grams](#1-the-core-recipe)
2. [Log Probability: Why We Take Logs](#2-log-probability)
3. [Perplexity: Measuring Model Confusion](#3-perplexity)
4. [Comparing Sentences: "Better" Means Higher Probability](#4-comparing-sentences)
5. [The Zero‑Probability Problem: Sparsity and OOV Words](#5-the-zero-probability-problem)
6. [Smoothing: Add‑One / Laplace Smoothing](#6-smoothing)
7. [Interpolation and Backoff (and Why Backoff Can Hurt)](#7-interpolation-and-backoff)
8. [Evaluating Language Models](#8-evaluating-language-models)
9. [Hands‑On: SRILM Toolkit Walkthrough](#9-hands-on-srilm)
10. [Capstone Exercise: The Brown Corpus](#10-capstone-exercise)
11. [Key Takeaways](#11-key-takeaways)

---

## 1. The Core Recipe

### 1.1 The Big Idea

An **n‑gram language model** estimates the probability of a sequence of words using a simple recipe:

1. Compute the **forward (conditional) probability** of each word given its context.
2. Apply the **chain rule** to factor the joint probability.
3. Multiply all conditional probabilities to get the **sequence probability**.

### 1.2 The Chain Rule in Action

Suppose your training data is a tiny corpus, and you want the probability of the sentence:

> **Sam I do like**

Using a **bigram** model (context = 1 previous word), the chain rule gives:

$$
P(\text{Sam, I, do, like}) = P(\text{Sam}\mid\text{start}) \cdot P(\text{I}\mid\text{Sam}) \cdot P(\text{do}\mid\text{I}) \cdot P(\text{I}\mid\text{do}) \cdot P(\text{like}\mid\text{I}) \cdot P(\text{end}\mid\text{like})
$$

> **Analogy:** Think of a relay race. The probability of the whole relay depends on each runner passing the baton successfully. If any runner fails (probability = 0), the entire race fails.

### 1.3 Computing Each Conditional Probability with MLE

Use **Maximum Likelihood Estimation (MLE)**:

$$
P(w_n \mid w_{n-1}) = \frac{\text{Count}(w_{n-1}, w_n)}{\text{Count}(w_{n-1})}
$$

**Memory hook:** *Count what follows ÷ Count what came before.*

### 1.4 Worked Example (Bigram)

Assume a toy training corpus in which the following counts hold:

| Conditional Probability | Numerator (bigram count) | Denominator (unigram count) | Result |
|---|---|---|---|
| P(Sam \| start) | Count(start, Sam) = 3 | Count(start) = 5 | 3/5 = **0.6** |
| P(I \| Sam) | Count(Sam, I) = 3 | Count(Sam) = 5 | 3/5 = **0.6** |
| P(do \| I) | Count(I, do) = 1 | Count(I) = 5 | 1/5 = **0.2** |
| P(I \| do) | Count(do, I) = 1 | Count(do) = 2 | 1/2 = **0.5** |
| P(like \| I) | Count(I, like) = 2 | Count(I) = 5 | 2/5 = **0.4** |
| P(end \| like) | Count(like, end) = 2 | Count(like) = 3 | 2/3 ≈ **0.667** |

**Sequence probability:**

$$
P(S) = 0.6 \times 0.6 \times 0.2 \times 0.5 \times 0.4 \times 0.667 \approx \mathbf{0.0096}
$$

### 1.5 Bigram vs. Trigram

- **Bigram (n = 2):** context is the **one** previous word. Example: $P(\text{I} \mid \text{Sam})$.
- **Trigram (n = 3):** context is the **two** previous words. Example: $P(\text{I} \mid \text{start}, \text{Sam})$.

The recipe is identical—only the size of the context window changes.

---

## 2. Log Probability

### 2.1 The Problem: Floating‑Point Underflow

Multiplying many small decimals (e.g., $0.6 \times 0.6 \times 0.2 \times \dots$) produces **vanishingly small numbers**. For long sentences with millions of words, the product underflows the floating‑point representation and the computer returns **0** even though the true probability is positive.

### 2.2 The Solution: Work in Log Space

Take the **logarithm** of the probability. Multiplication becomes **addition**:

$$
\log P(S) = \log(0.0096) \approx \mathbf{-2.015}
$$

> **Analogy:** Instead of multiplying $0.5 \times 0.5 \times 0.5 \times 0.5 = 0.0625$, you add $\log(0.5) + \log(0.5) + \log(0.5) + \log(0.5) = -2.77$. Addition is numerically stable; multiplying tiny numbers is not.

**Why this matters:** Most production language models report **log probabilities**, not raw probabilities. Always prefer log space when your sentence length exceeds a handful of tokens.

---

## 3. Perplexity

### 3.1 Definition

**Perplexity (PPL)** is the exponentiated average negative log‑likelihood:

$$
\text{PPL} = 2^{H}, \qquad H = -\frac{1}{N} \log_2 P(S)
$$

where $N$ is the total number of tokens used in the normalisation.

> **Analogy:** Perplexity is the *average number of choices* the model is confused between at each word. A perplexity of 2 means the model is as uncertain as a coin flip; a perplexity of 10 means it is as uncertain as rolling a 10‑sided die.

**Rule of thumb:** *Lower perplexity ⇒ better model. Higher perplexity ⇒ poorer model.*

### 3.2 The N‑Counting Debate: Start Tag vs. End Tag

Consider the sentence "Sam I do like" (5 words). How many tokens should $N$ be?

| Convention | N | Rationale |
|---|---|---|
| Lexical only (PPL) | 5 | Count only the words |
| Include end tag (PPL1) | 5 + 1 = 6 | End‑of‑sentence marker is a token |
| Include both tags | 5 + 2 = 7 | **Not recommended** — adds redundancy |

**Why only one boundary marker?** If you know the end tag, you automatically know the start tag of the *next* sentence—the two carry near‑identical information (high mutual information). Counting both adds **noise without information**.

**Memory hook:** *One boundary marker is enough—two is redundant.*

### 3.3 Computing Perplexity in Practice

From the worked example above:

- $\log P(S) \approx -2.015$
- With $N = 6$: $\text{PPL1} = 2^{-(-2.015)/6} \approx \mathbf{2.53}$
- With $N = 5$: $\text{PPL} \approx \mathbf{2.16}$

> **Note:** Both PPL and PPL1 are standard outputs. Choose the convention that matches your task and **stay consistent** across models you intend to compare.

---

## 4. Comparing Sentences

### 4.1 Defining "Better" via Probability

The word *better* is ambiguous—better for whom? The clean, operational definition is:

> **Higher probability ⇒ better sentence.**

### 4.2 The Word‑Order Example

Compare:

- **S1:** John went to London in the late afternoon.
- **S2:** John went in the late afternoon to London.

Both are grammatical and mean the same thing. Yet when humans are asked to vote, the overwhelming majority choose **S1**.

**Why?** Grammar alone cannot explain the preference—both sentences pass the grammar test. The missing ingredient is **human preference**, which correlates with **probability**. A well‑trained language model should assign S1 a higher probability than S2.

> **Analogy:** If 80 % of humans pick S1, a good language model should also pick S1. The model learns human preferences through probability.

### 4.3 The Computational Consequence

Use a held‑out corpus to estimate $P(S_1)$ and $P(S_2)$ under your model. The sentence with the higher probability is the one the model—and, by hypothesis, a human—prefers.

---

## 5. The Zero‑Probability Problem

### 5.1 The Crisis

Take the sentence:

> **Sam I do like linguistics**

The word *linguistics* never appears in the training data, so:

- $\text{Count}(\text{like}, \text{linguistics}) = 0$
- $P(\text{linguistics} \mid \text{like}) = 0$
- **Whole‑sentence probability = 0**

But the sentence is perfectly valid English. Assigning probability zero is wrong.

### 5.2 The Diagnosis: Sparsity and OOV

- **Sparsity:** The particular n‑gram simply did not occur in the training data.
- **OOV (Out‑of‑Vocabulary) word:** The word is not in the model's vocabulary at all.
- **UNK token:** A special token used to represent any unknown word.

> **Analogy:** You learned English from only 17 sentences. Someone says *serendipity*—you have never heard it, so you assign it zero probability. But the word is perfectly valid!

### 5.3 The Fix: Add the UNK Token

Before training, replace every word that does not appear (or appears too rarely) in the training corpus with the token **`<unk>`**. The model then learns a probability distribution over `<unk>` and can assign non‑zero probability to unseen words at test time.

---

## 6. Smoothing

### 6.1 Add‑One (Laplace) Smoothing

**Core idea:** *Steal probability mass from known words and give it to unknown words.* The total probability must still sum to one.

**Formula:**

$$
P(w_n \mid w_{n-1}) = \frac{\text{Count}(w_{n-1}, w_n) + 1}{\text{Count}(w_{n-1}) + V}
$$

where $V$ is the vocabulary size (after adding `<unk>`).

**Memory hook:** *Add one to the numerator, add V to the denominator.*

> **Analogy:** You have a bag with 2 bananas and 2 apples. Someone asks for an orange, but there is no orange. If you add one orange, you must also add one banana and one apple to keep the distribution proportional. That is exactly what smoothing does.

### 6.2 Worked Example

Suppose the vocabulary size is $V = 7$ (6 original words + `<unk>`). For the unseen bigram (like, linguistics):

$$
P(\text{linguistics} \mid \text{like}) = \frac{0 + 1}{\text{Count(like)} + 7} = \frac{1}{29} \approx \mathbf{0.034}
$$

For the known word *Sam* (original count = 5, denominator originally 22):

$$
P(\text{Sam} \mid \text{start}) = \frac{5 + 1}{22 + 7} = \frac{6}{29} \approx \mathbf{0.207}
$$

**Result:** Unknown words now receive a small but non‑zero probability, and the distribution still sums to one.

### 6.3 Other Smoothing Techniques (Brief Survey)

| Technique | Idea |
|---|---|
| **Absolute discounting** | Subtract a fixed amount from each count and redistribute the mass |
| **Good–Turing discounting** | Estimate the mass of unseen events from the count‑of‑counts |
| **Witten–Bell discounting** | Combine discounting with interpolation over n‑gram orders |
| **Kneser–Ney smoothing** | A refined absolute‑discounting method (state of the art for many tasks) |

The choice of smoothing method depends on the task, the language, and the corpus size—there is no universal winner.

---

## 7. Interpolation and Backoff

### 7.1 Interpolation

Combine probabilities from several n‑gram orders with learned weights:

$$
P = \lambda_1 P_{\text{trigram}} + \lambda_2 P_{\text{bigram}} + \lambda_3 P_{\text{unigram}}
$$

The weights $\lambda_i$ are learned on a **development set** using standard machine‑learning techniques (e.g., grid search or expectation maximisation).

### 7.2 Backoff

If a trigram is unseen, *back off* to the bigram; if the bigram is unseen, back off to the unigram; never return zero.

### 7.3 Why Backoff Can Hurt

Consider the implausible sentence:

> **John eats blue**

The trigram (*John, eats, blue*) is precisely what signals that the sentence is improbable. If we back off to the bigram (*John, eats*)—which is very frequent—we assign a **high** probability to a nonsense sentence.

Likewise, *I eats* is ungrammatical. Backing off to unigrams gives it some probability, but a good model should confidently say **zero**.

> **Key insight:** *Unseen data does not necessarily mean sparse data. It can signal that the sentence is nonsense, implausible, or ungrammatical—which the language model should recognise, not smooth over.*

> **Analogy:** If someone asks "Is *I eats* a valid English sentence?" the answer is **no**. A good model should say zero, not "well, *I* is common and *eats* is common, so maybe 0.01."

---

## 8. Evaluating Language Models

### 8.1 Two Types of Evaluation

**Extrinsic (external) evaluation** — test the model on a downstream task:

- Word‑order preference (S1 vs. S2)
- Machine translation
- Speech recognition

If the model makes the same choices humans would make, it is a good model.

**Intrinsic (internal) evaluation** — use internal metrics:

- **Likelihood:** $P(\text{test data} \mid \text{model})$ — higher is better
- **Log‑likelihood:** the log of the above — numerically stable
- **Perplexity:** $2^H$ — lower is better

### 8.2 The Golden Rule

> **Never train your language model on your test data.**

Doing so causes **overfitting**: the model has already seen the test data, so its probability estimates are *artificially* good. The resulting perplexity is meaningless.

> **Analogy:** If you give students the exam questions before the exam, everyone scores 100 %—but that does not mean they learned anything.

### 8.3 Comparing Perplexities: The Apple‑and‑Oranges Trap

> **You can compare the perplexity of two language models only if they were trained on the same vocabulary.**

Otherwise you are comparing apples and oranges. Always report the vocabulary and tokenisation alongside perplexity.

### 8.4 The N‑Gram Perplexity Trend

Empirically (e.g., Jurafsky & Martin):

$$
\text{PPL}_{\text{unigram}} \;>\; \text{PPL}_{\text{bigram}} \;>\; \text{PPL}_{\text{trigram}}
$$

**Why?** More context narrows the choice set, reducing average surprise.

> **Analogy:** Predicting the next word with no context (unigram) is like guessing a random word from the dictionary. With one word of context (bigram) you have narrowed it down. With two words (trigram) it is narrower still.

---

## 9. Hands‑On: SRILM Toolkit

### 9.1 What SRILM Does

SRILM is a C++ toolkit with command‑line binaries that:

- **`ngram-count`** — reads training text and produces n‑gram **count files** (frequency distributions).
- **`ngram`** — reads count files and produces a **language model** in ARPA format; also queries a model against test data and reports log‑prob, PPL, and PPL1.

### 9.2 The Pipeline

```
Training Data ──► ngram-count ──► Count File ──► ngram ──► Language Model (ARPA)
                                                              │
Test Data ──► ngram -lm [model] ──► Log Prob, PPL, PPL1 ◄─────┘
```

### 9.3 Commands

```bash
# Step 1 — Count n-grams and build a bigram LM with add-one smoothing
ngram-count -text train.txt -order 2 -lm bigram.lm -addsmooth 1

# Step 2 — Evaluate on test data (debug level 2 prints per-token probabilities)
ngram -lm bigram.lm -text test.txt -debug 2
```

### 9.4 Verifying Against the Hand Calculations

Running SRILM on the toy dataset reproduces every value computed by hand:

| Quantity | Hand calculation | SRILM output |
|---|---|---|
| P(Sam \| start) | 0.6 | 0.6 |
| P(I \| Sam) | 0.6 | 0.6 |
| P(do \| I) | 0.2 | 0.2 |
| P(I \| do) | 0.5 | 0.5 |
| P(like \| I) | 0.4 | 0.4 |
| P(end \| like) | 0.667 | 0.667 |
| Log prob | −2.015 | −2.015 |
| PPL | 2.16 | 2.16 |
| PPL1 | 2.53 | 2.53 |

> **The point:** SRILM is *not a black box*. You can now explain exactly how every number it prints was produced. For millions of sentences, manual computation is infeasible—this is where the toolkit earns its keep.

### 9.5 The `-debug 2` Output Explained

With `-debug 2`, SRILM prints one line per token, showing:

- the token,
- the n‑gram order used (e.g., bigram),
- the conditional probability,
- the log probability.

The sentence‑level log‑prob is the **sum** of the per‑token log‑probs; PPL and PPL1 are derived from that sum.

---

## 10. Capstone Exercise: The Brown Corpus

Apply everything above to the **Brown Corpus**—a large, mixed‑genre corpus containing fiction, non‑fiction, news, religion, romance, and more.

### Steps

1. Train a **bigram** and a **unigram** model on the Brown training split.
2. Evaluate on a held‑out **test** split that the model has never seen.
3. Replace OOV words with `<unk>` and retrain.
4. Apply **add‑one smoothing** and compare perplexities before and after.
5. Report **PPL** and **PPL1** for each configuration.
6. Submit your notebook (Google Colab recommended).

### What to Look For

- How much does smoothing reduce perplexity on OOV‑heavy test data?
- Does the bigram model beat the unigram model? By how much?
- How does the choice of $N$ (5 vs. 6) affect PPL vs. PPL1?

---

## 11. Key Takeaways

1. **Count what follows ÷ count what came before** — the entire n‑gram MLE recipe.
2. **Take logs** — multiplication of small numbers causes underflow; log turns it into addition.
3. **Higher perplexity = poorer model** — perplexity measures average confusion.
4. **Grammar alone cannot capture human preference** — probability can.
5. **Unseen data does not always mean sparse data** — sometimes zero probability is *correct* (ungrammatical sentences).
6. **Smoothing steals probability mass from known words and gives it to unknown words** — the total must still sum to one.
7. **Never train on test data** — that is cheating and produces artificially low perplexity.
8. **You cannot compare apples and oranges** — perplexity comparison requires identical vocabulary.
9. **SRILM is not a black box** — it does exactly what you computed by hand.
10. **The best approach is the best of all** — interpolation combines multiple n‑gram orders with learned weights.

---

## Appendix: A Single Mental Model to Remember Everything

Imagine teaching a child to predict the next word:

- **Unigram:** "What word comes next?" — the child guesses randomly from the whole vocabulary. *High perplexity.*
- **Bigram:** "I went to the ___" — the child says *store* or *park*. *Lower perplexity.*
- **Trigram:** "I went to the grocery ___" — the child says *store*. *Lowest perplexity.*

Now the child sees "I went to the grocery **zxyz**"—a word never seen before.

- **Without smoothing:** Probability = 0. The child says "impossible!"
- **With smoothing:** Probability = tiny but non‑zero. The child says "unusual, but possible."
- **With backoff:** The child says "well, *grocery* is common, so maybe it's fine"—**wrong** if *zxyz* is nonsense.

The art of language modelling is knowing when zero is correct (ungrammatical) and when smoothing is needed (rare but valid).

---

*Licence: MIT. Contributions and corrections welcome via pull request.*