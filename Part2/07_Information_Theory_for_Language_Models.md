# Information Theory for Language Models: A Complete Tutorial

## Introduction

Language is fundamentally about uncertainty. When you speak or write, each word you choose is one option among many possibilities. Information theory gives us the mathematical tools to quantify that uncertainty — and to understand how context reduces it.

This tutorial builds from the simplest possible examples to the concepts that power modern language models. By the end, you'll understand entropy, conditional entropy, mutual information, and KL divergence — and you'll see exactly how they connect to the way machines predict language.

---

## Part 1: Entropy — Measuring Uncertainty

### The Eight-Sided Die

Imagine rolling an eight-sided die. Before you roll, you don't know the outcome. After you roll, you do. How much information do you receive when you learn the result?

Assume all eight outcomes are equally likely. Each has probability 1/8. To transmit the result to someone else, you need to encode it as a binary message. How many bits do you need?

With 3 bits, you can represent 2³ = 8 distinct values. So 3 bits is sufficient.

This is exactly what entropy measures: **the expected length of the shortest message needed to transmit the outcome of a random variable**.

The formula is:

```
H(X) = -Σ p(x) log₂ p(x)
```

For the eight-sided die:

```
H(X) = -8 × (1/8) log₂(1/8)
     = -8 × (1/8) × (-3)
     = 3 bits
```

### The Fair Coin

Now toss a fair coin. There are two outcomes — heads and tails — each with probability 1/2.

```
H(X) = -[ (1/2) log₂(1/2) + (1/2) log₂(1/2) ]
     = -[ (1/2)(-1) + (1/2)(-1) ]
     = 1 bit
```

One bit is the minimum length needed to transmit the outcome. This makes intuitive sense: a coin flip is the simplest possible binary decision.

### The Probability-Entropy Curve

Entropy depends on the probability distribution. Consider a binary event with probability p:

```
Entropy
   ^
   |     *
   |    / \
   |   /   \
   |  /     \
   | /       \
   |/         \
   +-------------> p
   0    0.5    1
```

- **When p = 0 or p = 1**: Entropy is zero. The outcome is certain — there is no uncertainty. Mathematically, 0 × log(0) is treated as 0, and 1 × log(1) = 0.
- **When p = 0.5**: Entropy is maximum. The outcome is completely uncertain — you have no information to predict it.

The intuition: entropy measures surprise. If you already know what will happen, learning the outcome gives you no new information. If the outcome is completely random, learning it gives you maximum information.

### The Critical Limitation: Entropy Ignores History

Here is where things get interesting for language. The entropy we just computed treats each event independently. It says nothing about sequences.

Consider the sentence: *"A small oblong reading lamp on the desk."*

- Without context, predicting the next word after "The" is nearly impossible — there are thousands of possibilities.
- After "The small oblong reading," the choices narrow dramatically.
- After "The small oblong reading lamp on the," the next word is almost certainly "desk."

As more context is provided, the uncertainty decreases. But simple entropy doesn't capture this — it treats each word in isolation.

**Language depends on history.** To model language, we need tools that account for context. That's where conditional entropy comes in.

---

## Part 2: Joint Entropy — Two Variables Together

When we have two random variables, X and Y, we can ask: how much information is needed to specify both their values simultaneously?

The joint entropy is:

```
H(X, Y) = -Σ Σ p(x, y) log₂ p(x, y)
```

**Example:** Consider the words "New" and "Delhi."

- H("New") alone might be 4 bits — many words could follow.
- H("Delhi") alone might be 8 bits — many cities exist.
- H("New", "Delhi") together is less than 12 bits because these words are correlated. If you know one, you have information about the other.

Joint entropy captures the uncertainty of both variables at once. When variables are correlated, their joint entropy is less than the sum of their individual entropies.

---

## Part 3: Conditional Entropy — The Heart of Language Modeling

### Definition

Conditional entropy H(Y|X) measures: **how much extra information is needed to communicate Y, given that you already know X?**

In language terms: how uncertain is the next word, given the words that came before?

### The Umbrella Example

Let X be the weather (sunny, rainy, cloudy) and Y be whether you carry an umbrella.

| Weather (X) | P(Umbrella = Yes \| X) | Entropy of Y given X |
|-------------|------------------------|----------------------|
| Sunny | 0.1 | Low — you almost certainly won't carry one |
| Rainy | 0.9 | Low — you almost certainly will carry one |
| Cloudy | 0.5 | High — genuinely uncertain |

The conditional entropy is the **weighted average** of these individual entropies:

```
H(Y|X) = Σ p(x) × H(Y|X=x)
```

This is the key formula. Conditional entropy averages the uncertainty of Y across all possible values of X, weighted by how likely each X is.

### The Full Derivation

Starting from the definition:

```
H(Y|X) = Σ p(x) H(Y|X=x)
       = Σ p(x) [ -Σ p(y|x) log₂ p(y|x) ]
       = -Σ Σ p(x) p(y|x) log₂ p(y|x)
```

Using the identity p(x, y) = p(x) p(y|x):

```
H(Y|X) = -Σ Σ p(x, y) log₂ p(y|x)
```

This is the standard form you'll see in textbooks. Note that the joint probability p(x, y) appears because we're averaging over all combinations of X and Y.

### Conditioning Reduces Entropy

The fundamental insight: **knowing X reduces uncertainty about Y**.

```
Before knowing X:  H(Y)      = high uncertainty
After knowing X:   H(Y|X)    = lower uncertainty
```

Without weather information, your umbrella decision has entropy of about 1 bit (50-50 chance). With weather information, entropy drops to about 0.3 bits — because weather tells you what to do.

This is why context matters in language. Each additional word you know reduces the entropy of the next word.

---

## Part 4: The Chain Rule — Building Up Complexity

Joint entropy can be decomposed using the chain rule:

```
H(X, Y) = H(X) + H(Y|X)
```

**Derivation:**

```
H(X, Y) = -Σ Σ p(x, y) log₂ p(x, y)
```

Apply Bayes' rule: p(x, y) = p(x) p(y|x).

```
H(X, Y) = -Σ Σ p(x, y) [ log₂ p(x) + log₂ p(y|x) ]
       = -Σ Σ p(x, y) log₂ p(x) - Σ Σ p(x, y) log₂ p(y|x)
       = H(X) + H(Y|X)
```

### The Sentence Example

Consider the sentence: *"John went to Paris."*

The joint entropy of all four words can be decomposed:

```
H(John, went, to, Paris) = H(John)
                         + H(went | John)
                         + H(to | John, went)
                         + H(Paris | John, went, to)
```

Each term represents the additional uncertainty of the next word given everything before it. As context grows, each term shrinks — "Paris" after "John went to" is much more predictable than "Paris" in isolation.

**Symmetry:** The chain rule works in either direction:

```
H(X, Y) = H(X) + H(Y|X) = H(Y) + H(X|Y)
```

This symmetry is fundamental — and it leads directly to mutual information.

---

## Part 5: Mutual Information — Quantifying Shared Information

### Intuition

Mutual information measures: **how much does knowing one variable reduce uncertainty about the other?**

Formally, the mutual information between X and Y is:

```
I(X; Y) = H(X) - H(X|Y)
```

This is the reduction in uncertainty about X that comes from knowing Y.

By symmetry:

```
I(X; Y) = H(Y) - H(Y|X) = I(Y; X)
```

### The Probability Formula

Mutual information can also be written directly in terms of probabilities:

```
I(X; Y) = Σ Σ p(x, y) log₂ [ p(x, y) / (p(x) p(y)) ]
```

The ratio inside the log compares the joint probability to what it would be if X and Y were independent. If they're independent, p(x, y) = p(x) p(y), the ratio is 1, and the log is 0 — mutual information is zero. The more dependent they are, the larger the ratio and the larger the mutual information.

### The Venn Diagram

Mutual information has a beautiful geometric interpretation:

```
    H(X)           H(Y)
   ┌─────┐       ┌─────┐
   │     │       │     │
   │  ┌──┼───────┼──┐  │
   │  │  │ I(X;Y)│  │  │
   │  └──┼───────┼──┘  │
   │     │       │     │
   └─────┘       └─────┘
```

- The left circle is H(X), the right is H(Y).
- The intersection is I(X; Y) — the information shared between them.
- The left-only region is H(X|Y) — uncertainty about X that remains after knowing Y.
- The right-only region is H(Y|X) — uncertainty about Y that remains after knowing X.

### The Student-Teacher Example

Let X be whether a student studies, and Y be whether the student passes.

- If studying and passing are completely unrelated (hypothetically), knowing Y tells you nothing about X. Mutual information is zero.
- In reality, almost everyone who passes studied, and almost everyone who fails did not study. Observing Y gives substantial information about X. Mutual information is high.

### The "New Delhi" Example

In language, mutual information identifies words that occur together more often than chance would predict.

Consider "New" and "Delhi":

- P(Delhi | New) is much higher than P(Delhi) alone. Knowing the previous word is "New" substantially changes the probability of the next word.
- Therefore, "New" and "Delhi" have high mutual information.

This is how we identify **collocations** — word pairs that habitually appear together:

- "New Delhi" (compound noun)
- "Fast food" (fixed phrase)
- "To do" (common pairing)

Contrast with **associations** — words that can appear together but aren't fixed:

- "New car" (perfectly valid, but not a fixed unit)
- "Fast water" (grammatically possible, but unusual)

Mutual information helps distinguish collocations from mere associations. Words with high mutual information are candidates for being treated as units in language models.

### Does More Context Always Help?

A natural question: if adding context reduces uncertainty, does adding more context always help?

Not necessarily. It depends on the **type of words** in the context.

- Adding "the" provides little information — it's highly predictable and doesn't constrain the next word much.
- Adding a content word like "Paris" provides substantial information — it strongly constrains what can follow.
- Adding irrelevant context can even hurt — it adds noise without reducing uncertainty.

The quality of context matters as much as the quantity.

---

## Part 6: Relative Entropy (KL Divergence) — Comparing Distributions

### Definition

Relative entropy, also called Kullback-Leibler divergence, measures how one probability distribution differs from another.

For two probability distributions P and Q over the same event space:

```
D(P || Q) = Σ p(x) log₂ [ p(x) / q(x) ]
```

### Intuition: Wasted Bits

KL divergence has a precise interpretation: **the average number of extra bits needed to encode events from distribution P using a code optimized for distribution Q instead of P.**

Imagine you're packing for a trip. You check the weather forecast for New York (distribution Q) and pack accordingly. But you're actually going to India (distribution P). The "wasted bits" are the extra items you packed that you don't need, or the items you didn't pack that you do need.

### Key Properties

1. **Not symmetric:** D(P || Q) ≠ D(Q || P). The divergence from P to Q is different from Q to P.
2. **Non-negative:** D(P || Q) ≥ 0, with equality if and only if P = Q.
3. **Same event space required:** You can only compare distributions over the same set of outcomes. Comparing distributions over different event spaces is "apples and oranges."

### Why This Matters for Language Models

In practice, the distribution a model learns from training data may not match the distribution it encounters in deployment.

- A model trained on New York Times articles (formal English) may perform poorly on Twitter data (informal English).
- A model trained on American English may struggle with Indian English.

KL divergence quantifies this mismatch. It tells you how much "extra work" the model does because its learned distribution doesn't match reality.

---

## Part 7: All Concepts Together

Here's how everything connects:

```
                    ENTROPY H(X)
                         │
         ┌───────────────┼───────────────┐
         │               │               │
    JOINT H(X,Y)   CONDITIONAL H(Y|X)  RELATIVE D(P||Q)
         │               │               │
         └───────┬───────┘               │
                 │                       │
           CHAIN RULE                    │
        H(X,Y)=H(X)+H(Y|X)               │
                 │                       │
                 ▼                       │
         MUTUAL INFORMATION I(X;Y) ◄─────┘
         I(X;Y) = H(X) - H(X|Y)
                = H(Y) - H(Y|X)
```

The throughline: all these concepts answer the same fundamental question — **how much information is in language, and how do we measure it?**

---

## Part 8: Activities

### Activity 1: Entropy and Sequences

**Question:** Explain the concept of entropy. What does it say about a given symbol sequence?

**Answer:** Entropy measures the expected surprise per symbol. The more uncertain the next symbol, the higher the entropy. However, simple entropy says nothing about the sequence itself — it treats each symbol independently. To capture the randomness of a sequence, you need conditional entropy, which accounts for history and context.

### Activity 2: Proving Symmetry

**Question:** Prove that I(X; Y) = I(Y; X).

**Proof:**

```
I(X; Y) = Σ Σ p(x, y) log₂ [ p(x, y) / (p(x) p(y)) ]
```

Since p(x, y) = p(y, x), p(x) p(y) = p(y) p(x), and summation is commutative:

```
I(X; Y) = Σ Σ p(y, x) log₂ [ p(y, x) / (p(y) p(x)) ]
        = I(Y; X)
```

The symmetry follows directly from the symmetry of the joint probability and the commutativity of multiplication.

---

## Part 9: What's Next — Perplexity and Language Modeling

These concepts form the foundation for everything that follows in language modeling:

- **Perplexity** = 2^H(X). It's a more intuitive way to express entropy. If H(X) = 3 bits, perplexity = 8 — the model is as confused as choosing among 8 equally likely options. Lower perplexity means a better model.

- **N-gram models** use conditional entropy directly: P(word_n | word_1, ..., word_{n-1}). The chain rule tells us how to decompose the joint probability of a sentence into conditional probabilities.

- **Cross-entropy loss**, the standard training objective for language models, is directly related to KL divergence. Minimizing cross-entropy is equivalent to minimizing the divergence between the model's distribution and the true distribution.

---

## Summary

| Concept | Formula | Intuition |
|---------|---------|-----------|
| Entropy | H(X) = -Σ p(x) log p(x) | Expected surprise of a single variable |
| Joint entropy | H(X,Y) = -ΣΣ p(x,y) log p(x,y) | Surprise of two variables together |
| Conditional entropy | H(Y\|X) = -ΣΣ p(x,y) log p(y\|x) | Surprise of Y given X |
| Chain rule | H(X,Y) = H(X) + H(Y\|X) | Decompose joint entropy |
| Mutual information | I(X;Y) = H(X) - H(X\|Y) | Shared information between variables |
| KL divergence | D(P\|\|Q) = Σ p(x) log [p(x)/q(x)] | Extra bits from using wrong distribution |

---

## Final Thought

Every time you use a language model, these concepts are running underneath. Entropy decreases as context builds. Mutual information guides predictions. KL divergence ensures the model's distribution matches reality. The mathematics in this tutorial is the grammar of how machines understand language — and now you speak it fluently.