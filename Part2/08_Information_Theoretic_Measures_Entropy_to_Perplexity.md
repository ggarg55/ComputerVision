# Information-Theoretic Measures for Language: From Entropy to Perplexity

## A Self-Contained Tutorial on Measuring Uncertainty in Language

---

## Introduction

How do we quantify information in language? How do we measure whether a language model is good? And why are humans so remarkably good at predicting what someone will say next?

This tutorial answers these questions by connecting **information theory** to **language modeling**. We'll start with the basics of entropy and conditional entropy, apply them to words instead of weather, derive **perplexity** (the most widely used metric for language models), and then build simple **N-gram language models** from scratch using nothing but counting.

By the end, you'll be able to:
- Compute conditional entropy and pointwise surprisal for words
- Explain what a language model *actually* is
- Derive perplexity from first principles
- Build unigram and bigram models by counting from a corpus
- Represent language as a probabilistic finite state machine
- Explain why humans can predict each other's words

Let's begin.

---

## Part 1: Conditional Entropy — Reducing Uncertainty

### The Core Idea

**Conditional entropy** measures how much uncertainty remains about a random variable **Y** once you already know the value of another random variable **X**.

Formally:

$$
H(Y \mid X) = \sum_{x} P(x) \cdot H(Y \mid X = x)
$$

We average the entropy of Y over all possible values of X, weighted by how likely each X is.

### A Concrete Example: Weather and Umbrellas

Let:
- **X** = weather (sunny or rain)
- **Y** = whether I carry an umbrella

| Weather (X) | P(X) | P(umbrella = yes \| X) | H(Y \| X = x) |
|---|---|---|---|
| Sunny | 0.7 | 0.1 | 0.47 bits |
| Rain | 0.3 | 0.9 | 0.47 bits |

$$
H(Y \mid X) = 0.7 \times 0.47 + 0.3 \times 0.47 = \mathbf{0.47 \text{ bits}}
$$

Compare this to the entropy of Y alone (≈ 0.88 bits). **Knowing the weather cut our uncertainty nearly in half.**

**Key insight:** When X tells you something about Y, conditioning reduces entropy:

$$
H(Y \mid X) \leq H(Y)
$$

### The Derivation You Should Know

Conditional entropy can be written in several equivalent forms:

$$
H(Y \mid X) = H(X, Y) - H(X)
$$

$$
= -\sum_{x} \sum_{y} P(x, y) \log_2 P(y \mid x)
$$

Understanding this derivation matters. It connects joint entropy, marginal entropy, and conditional entropy into a single coherent picture.

---

## Part 2: Conditional Entropy Applied to Language

### Words Instead of Weather

In language, our random variables become **words**. We care about the entropy of a word **given its prior context** (the history of words before it).

Consider the phrase:

> "peanut butter and ___"

- **History (X):** "peanut butter and"
- **Next word (Y):** "jelly" or "bread" or something else

The structure mirrors our weather/umbrella example exactly:
- X = context (history of words)
- Y = next word

### Pointwise Conditional Entropy (Surprisal)

Conditional entropy averages over **all possible** contexts. **Pointwise conditional entropy** focuses on **one specific** context:

$$
\text{Surprisal}(w \mid \text{history}) = -\log_2 P(w \mid \text{history})
$$

**Worked example:**

| Word | P(word \| "he put the coffee on the") | Surprisal (bits) |
|---|---|---|
| table | 0.6 | −log₂(0.6) ≈ **0.74** |
| desk | 0.3 | −log₂(0.3) ≈ **1.74** |
| helicopter | 0.001 | −log₂(0.001) ≈ **9.97** |

The word "helicopter" is far more surprising than "table" — and surprisal quantifies exactly how much.

**Log base matters:**
- **log₂** → measured in **bits**
- **log₁₀** → measured in **Hartleys**

---

## Part 3: Where Do Probabilities Come From?

### Corpora and Estimation

To compute P(jelly | peanut butter and), we need data. That data comes from a **corpus** — a large collection of text (millions or billions of words) that represents the language.

**The fundamental distinction:**

| | True Probability | Estimated Probability |
|---|---|---|
| What it is | The ultimate truth about language | What we compute from a corpus |
| Can we know it? | No — human limitation | Yes, approximately |
| Assumption | — | Our corpus is a good enough representative |

We assume that a sufficiently large corpus approximates the true distribution well enough that:

$$
P_{\text{estimated}}(w \mid \text{history}) \approx P_{\text{true}}(w \mid \text{history})
$$

**Example:** In a 1-billion-word corpus, "jelly" appears after "peanut butter and" 4,000 times, while "peanut butter and" appears 10,000 times total.

$$
\hat{P}(\text{jelly} \mid \text{peanut butter and}) = \frac{4000}{10000} = 0.4
$$

---

## Part 4: What Is a Language Model?

### The Definition

> **A language model is a probability distribution over language.**

That's it. When you "train a language model," you are:
1. Assuming your corpus represents the language
2. Estimating probabilities from that corpus

A language model is a **probability machine**: feed it a history, and it returns a probability for the next word.

### The Connection to Entropy

With a language model **M**, pointwise conditional entropy becomes:

$$
-\log_2 M(w \mid \text{history})
$$

Replace M with the true probability P, and you recover the theoretical formula from information theory.

---

## Part 5: Total Entropy of a Sequence — The Chain Rule

### Building Up a Sentence

For a sequence of words w₁ w₂ … wₙ, the **chain rule of entropy** gives us:

$$
H(w_1, w_2, \ldots, w_n) = H(w_1) + H(w_2 \mid w_1) + H(w_3 \mid w_1, w_2) + \cdots + H(w_n \mid w_1, \ldots, w_{n-1})
$$

Each word's entropy is conditioned on **everything that came before it**.

### Worked Example

Consider the sentence:

> **"he put the coffee on the table"**

The total surprisal of this sequence is:

$$
-\log_2 \big[ P(\text{he}) \cdot P(\text{put} \mid \text{he}) \cdot P(\text{the} \mid \text{he put}) \cdot P(\text{coffee} \mid \text{he put the}) \cdot P(\text{on} \mid \text{he put the coffee}) \cdot P(\text{table} \mid \text{he put the coffee on}) \big]
$$

**Total surprisal** (also called total entropy when we take the expectation) is the negative log probability of the entire sequence.

**Memory hook:** The chain rule is a relay race. Each word hands the baton (its probability) to the next, conditioned on everyone who ran before.

---

## Part 6: Entropy Rate — Surprisal Per Word

### Normalizing by Length

A long sentence naturally has higher total surprisal than a short one. To compare sequences of different lengths, we compute the **entropy rate**:

$$
\text{Entropy rate} = \frac{\text{Total entropy}}{n}
$$

where n is the number of tokens (words).

**Example:** If a 7-token sentence has total surprisal of 14 bits, the entropy rate is **2 bits per word**. On average, each word costs 2 bits to transmit.

**Memory hook:** Total entropy is the grocery bill; entropy rate is the price per item.

---

## Part 7: Perplexity — The Star Metric

### The Definition

**Perplexity** is the most widely used metric for evaluating language models. It is defined as:

$$
\text{Perplexity} = 2^{H}
$$

where H is the **entropy rate** (per-word entropy).

For a full sequence:

$$
\text{Perplexity} = M(w_1, w_2, \ldots, w_n)^{-1/n}
$$

### The Derivation (Step by Step)

Starting from total surprisal:

$$
\text{Total surprisal} = -\log_2 M(w_1, \ldots, w_n)
$$

Divide both sides by n:

$$
\frac{1}{n} \cdot (-\log_2 M(w_1, \ldots, w_n))
$$

Apply the log property a·log(b) = log(bᵃ):

$$
= -\log_2 \left( M(w_1, \ldots, w_n)^{1/n} \right)
$$

Exponentiate both sides with base 2:

$$
2^{\text{Entropy rate}} = M(w_1, \ldots, w_n)^{-1/n}
$$

So:

$$
\boxed{\text{Perplexity} = 2^{H} = M(w_1, \ldots, w_n)^{-1/n}}
$$

### What Perplexity Actually Means

Perplexity tells you the **effective number of choices** the model is juggling at each step.

**The branching-factor intuition:**

If at every step the model is choosing uniformly among **10** equally likely words:

$$
H = \log_2 10 \approx 3.32 \text{ bits}
$$

$$
\text{Perplexity} = 2^{3.32} = \mathbf{10}
$$

Perplexity literally equals the effective number of next-word choices.

### Interpreting Perplexity

| Property | Meaning |
|---|---|
| **Lower perplexity** | Better language model — more confident, fewer choices |
| **Higher perplexity** | Worse model — more confused, more choices |
| **Lower entropy** | Better prediction — more deterministic |
| **Higher entropy** | Worse prediction — more random |

### Entropy vs. Perplexity: Related but Not Identical

They are **proportional** (PP = 2^H), but they capture different essences:

- **Entropy** measures the minimum number of bits required to transmit a message / expected surprise.
- **Perplexity** measures the number of options per word.

**Memory hook:** Perplexity = "how many words is my model juggling?" A confident model juggles few; a confused model juggles many.

---

## Part 8: Why Are We Able to Predict Each Other?

This is a deep question. Three answers:

### 1. Shared Context and Structure

We speak the same language, so structures exist in **both** the speaker's and listener's minds. The predictability is common to both parties.

### 2. Language Structure Is Inherently Predictable

**Zipf's law** tells us:
- Short words are frequent and critical
- Long words are less frequent and less predictable
- The structure of language itself makes prediction possible

Interestingly, similar Zipf-like patterns appear in **non-human communication** too — whale songs show comparable statistical regularities.

### 3. More Context → Fewer Choices → Lower Entropy

As we give more history of words, the number of plausible next words **decreases**. Entropy decreases for both speaker and listener, making mutual prediction easier.

### The Limitation of Text

Text is "black and white" — it doesn't capture **emotion**, **tone**, or **sentiment**. When text alone is unpredictable, adding other modalities (speech, video, facial expressions) **further reduces entropy**. Context, genre, and domain also shape predictions.

**Memory hook:** Prediction works because we share structure. Text is black-and-white; emotions need more modalities.

---

## Part 9: N-gram Language Models — The Mechanics

### What Is N?

**N** = the number of words in the chunk:

| N | Name | Context Window |
|---|---|---|
| 1 | Unigram | 0 (no context) |
| 2 | Bigram | 1 previous word |
| 3 | Trigram | 2 previous words |
| 4 | Quadrigram | 3 previous words |

### The Markov Assumption

> The probability of a word depends on a **fairly recent, finite** number of previous words.

- Bigram → context window = 1
- Trigram → context window = 2

We don't look at the entire history — just a fixed window.

### Unigram Model

Words are treated as **independent**:

$$
P(\text{he put the coffee on the table}) = P(\text{he}) \cdot P(\text{put}) \cdot P(\text{the}) \cdot P(\text{coffee}) \cdot P(\text{on}) \cdot P(\text{the}) \cdot P(\text{table})
$$

No conditioning — just unconditional probabilities multiplied together.

### Bigram Model

Each word is conditioned on the **immediately preceding** word:

$$
P(\text{he put the coffee on the table}) = P(\text{put} \mid \text{he}) \cdot P(\text{the} \mid \text{put}) \cdot P(\text{coffee} \mid \text{the}) \cdot P(\text{on} \mid \text{coffee}) \cdot P(\text{the} \mid \text{on}) \cdot P(\text{table} \mid \text{the})
$$

### The Counting Formula

For a bigram:

$$
\boxed{P(w_n \mid w_{n-1}) = \frac{\text{Count}(w_{n-1}, w_n)}{\text{Count}(w_{n-1})}}
$$

**In words:** The probability of word wₙ given the previous word wₙ₋₁ equals the number of times the pair appears together, divided by the number of times the first word appears at all.

**Worked example:**
- Count("he", "put") = 100
- Count("he") = 200
- P(put | he) = 100 / 200 = **0.5**

### Trigram Model

$$
P(w_n \mid w_{n-2}, w_{n-1}) = \frac{\text{Count}(w_{n-2}, w_{n-1}, w_n)}{\text{Count}(w_{n-2}, w_{n-1})}
$$

**Worked example:**

$$
P(\text{coffee} \mid \text{he put}) = \frac{\text{Count}(\text{he put coffee})}{\text{Count}(\text{he put})}
$$

### Why Bayes' Rule Underlies This

We're applying Bayes' rule:

$$
P(x \mid y) = \frac{P(x, y)}{P(y)}
$$

For a trigram, all three words must co-occur as a triplet for the joint count. The denominator counts how often the two-word context appears.

### Computing Counts with Regular Expressions

Given a corpus of millions of sentences, use **grep** or **regular expressions**:

```bash
# Numerator: count of "put the coffee"
grep -c "put the coffee" corpus.txt

# Denominator: count of "put the"
grep -c "put the" corpus.txt
```

Divide and you have your trigram probability.

### Handling Sentence Boundaries

Treat `<s>` (start) and `</s>` (end) as **tokens**. Include them in your vocabulary and count them. This changes your total token count **n** and ensures every sentence has proper context at the beginning and end.

### Order Matters

In co-occurrence counting, **"I am" ≠ "am I"**. We compute ordered sequences throughout.

---

## Part 10: Finite State Machines as Language Representations

### The Graph View

Another way to represent language probabilistically: **finite state automata** where:
- **Nodes** = words
- **Edges** = transitions with probabilities

### Example

Consider a simple FSM:

```
        0.6
  "he" ------> "is"
   |  ↺        
   |  self-loop
   ↓ 0.4
  (other)
```

- P(he → is) = **0.6**
- The self-loop on "he" models repetition: "he he he is my brother"
- Outgoing transitions from a node sum to appropriate probabilities

### The Big Picture: Representing Language Structure

The question "How can we computationally represent the structure of language?" has multiple answers:

1. **Probability distributions** (language models)
2. **Finite state machines / graph-based representations**
3. **Context grammars** (future topic)
4. **Dependency trees** (future topic)

Each captures a different aspect of linguistic structure.

> **Note:** The self-loop in an FSM is a point of debate. In formal automata theory, self-loops have specific interpretations. In probabilistic language modeling, they represent repetition naturally. Context matters.

---

## Part 11: Worked Activity — Comparing Sentences

### The Setup

Training data:

```
<s> Sam slick </s>
<s> machine ... </s>
```

### Task 1: Unigram Model

Compute:

$$
P(\text{Sam}) \cdot P(\text{slick}) \quad \text{vs.} \quad P(\text{machine}) \cdot P(\ldots)
$$

Count each token (including `<s>` and `</s>`) divided by total tokens.

### Task 2: Bigram Model

Compute:

$$
P(\text{Sam} \mid \langle s \rangle) \cdot P(\text{slick} \mid \text{Sam}) \cdot P(\langle /s \rangle \mid \text{slick})
$$

$$
\text{vs.} \quad P(\text{machine} \mid \langle s \rangle) \cdot P(\ldots \mid \text{machine}) \cdot P(\langle /s \rangle \mid \ldots)
$$

### The Answer

The sentence with the **higher probability** under the model is the one the model finds more likely. Under a bigram model, the second sentence typically wins because its transitions are more probable given the training data.

**Key takeaway:** Higher probability = better sentence according to the model. This is how language models rank candidate sentences.

---

## Part 12: Summary Card

| Concept | Formula / Idea | Memory Hook |
|---|---|---|
| Conditional entropy | H(Y\|X) = Σ P(x)H(Y\|x) | Weather shrinks umbrella uncertainty |
| Pointwise surprisal | −log₂ P(word\|history) | One specific moment |
| Corpus estimation | count ÷ count; estimated ≈ true | Corpus is language's mirror |
| Language model | Probability distribution over language | Probability machine |
| Chain rule | H(w₁…wₙ) = Σ H(wᵢ\|w₁…wᵢ₋₁) | Relay race of probabilities |
| Entropy rate | Total entropy ÷ n | Price per item |
| Perplexity | PP = 2^H = M(w₁…wₙ)^(−1/n) | Number of words juggled |
| Bigram probability | Count(wₙ₋₁, wₙ) / Count(wₙ₋₁) | Bayes + corpus witness |
| Trigram probability | Count(wₙ₋₂, wₙ₋₁, wₙ) / Count(wₙ₋₂, wₙ₋₁) | Three-word pattern |
| Markov assumption | Finite recent context | Short memory window |
| FSM representation | Nodes = words, edges = probs | Language as a graph |
| Why we predict | Shared structure + context + Zipf | Structure lives in both minds |
| Text's limitation | No emotion → need speech/video | Black-and-white words |
| Order matters | "I am" ≠ "am I" | Sequences are ordered |
| `<s>` and `</s>` | Count as tokens | Boundaries are words too |

---

## Part 13: The Complete Arc

This tutorial follows one coherent arc:

```
Uncertainty → Measurement → Model → Evaluation
```

1. **Entropy** measures uncertainty.
2. **Conditional entropy** localizes uncertainty given context.
3. **Chain rule** assembles a sentence from conditional pieces.
4. **Entropy rate** normalizes by length.
5. **Perplexity** translates bits into "number of choices."
6. **N-grams** are the simplest probability machines.
7. **FSMs** offer a graph-based view.
8. **Human prediction** grounds it all in shared cognitive structure.

---

## Exercises

1. **Conditional entropy by hand.** Given a tiny corpus `["the cat sat", "the dog sat", "the cat ran"]`, compute H(next word | "the") and H(next word | "the cat"). Show your work.

2. **Perplexity derivation.** Starting from the definition PP = 2^H, derive the sequence-level formula PP = M(w₁…wₙ)^(−1/n) step by step.

3. **Bigram counting.** Using the corpus `["he put the coffee", "he put the tea", "she put the coffee"]`, compute:
   - P(put | he)
   - P(coffee | the)
   - P(the | put)

4. **Unigram vs. bigram.** For the same corpus, compute the unigram probability and bigram probability of "he put the coffee". Which is higher? Why?

5. **FSM design.** Draw a finite state machine that generates the language `{"he is", "he he is", "he he he is"}` with appropriate probabilities.

6. **Conceptual question.** Explain in your own words why lower perplexity means a better language model. Use the "number of choices" intuition.

---

## Further Reading

- Jurafsky & Martin, *Speech and Language Processing*, **Section 3.7** — for the approximations involved in N-gram models
- Cover & Thomas, *Elements of Information Theory* — for rigorous derivations of entropy, conditional entropy, and the chain rule
- Zipf, *Human Behavior and the Principle of Least Effort* — for the empirical laws of word frequency

---

*This tutorial is released under CC BY 4.0. Contributions and corrections welcome via pull request.*