# Information Theory & Language: From Zipf's Law to Shannon's Entropy

A self-contained tutorial on how language carries information — and how we measure it.

---

## Table of Contents

1. [Why This Matters](#1-why-this-matters)
2. [Zipf's Law: The Uneven Distribution of Words](#2-zipfs-law-the-uneven-distribution-of-words)
3. [Why Zipf Exists: Least Effort and the Efficiency–Robustness Trade-off](#3-why-zipf-exists-least-effort-and-the-efficiencyrobustness-trade-off)
4. [The Monkey-Typing Experiment](#4-the-monkey-typing-experiment)
5. [Redundancy: Q Is Always Followed by U](#5-redundancy-q-is-always-followed-by-u)
6. [Language as a Probabilistic System](#6-language-as-a-probabilistic-system)
7. [Shannon's Game: N-Order Approximations](#7-shannons-game-n-order-approximations)
8. [The Limits of N-Grams](#8-the-limits-of-n-grams)
9. [Entropy: Measuring Uncertainty](#9-entropy-measuring-uncertainty)
10. [Redundancy in English: The 2.3-Bit Result](#10-redundancy-in-english-the-23-bit-result)
11. [Entropy Is Not Sequence Randomness](#11-entropy-is-not-sequence-randomness)
12. [Efficiency vs. Robustness in Communication](#12-efficiency-vs-robustness-in-communication)
13. [Summary and Key Takeaways](#13-summary-and-key-takeaways)

---

## 1. Why This Matters

Every large language model you've ever used — GPT, Claude, Gemini — is built on a single foundational idea: **language is probabilistic**. The next word is not determined; it is *likely*. Understanding *how likely*, and *how much information* each word carries, is the bedrock of natural language processing.

This tutorial answers four questions:

- **What patterns exist in language?** (Zipf's Law)
- **Why do those patterns exist?** (Least effort, efficiency vs. robustness)
- **How can we model language probabilistically?** (Shannon's n-order approximations)
- **How do we measure the information in language?** (Entropy)

By the end, you'll understand why English is roughly **50% redundant**, why a fair coin carries exactly **1 bit** of information, and why your brain can spot a typo before you consciously read it.

---

## 2. Zipf's Law: The Uneven Distribution of Words

### The Observation

If you count how often each word appears in any large text — a novel, a newspaper corpus, a chat log — you'll find a strikingly uneven distribution:

- A **few words are extremely frequent** ("the", "a", "of", "and")
- **Most words are extremely rare** (appearing once or twice in a million words)

This is not unique to English. It holds across languages, across genres, across centuries.

### The Law

Zipf's Law states that a word's frequency is **inversely proportional to its rank**:

\[
\text{frequency} \times \text{rank} \approx \text{constant}
\]

If the most frequent word appears \(f\) times, the second most frequent appears roughly \(f/2\) times, the third \(f/3\), and so on.

### A Simple Example

Imagine a corpus where "the" (rank 1) appears 1,000 times:

| Rank | Word | Expected Frequency |
|------|------|-------------------|
| 1 | the | 1,000 |
| 2 | of | 500 |
| 10 | — | 100 |
| 100 | — | 10 |
| 1,000 | — | 1 |

### The City-Population Analogy

Think of city populations: a handful of megacities (Tokyo, Delhi, Shanghai) hold tens of millions, while thousands of small towns hold a few thousand each. Zipf's Law is the "megacity law" of words — **few giants, many dwarfs**.

### Why It Looks Like a "Long Tail"

Plot frequency against rank on a log-log scale and you get a nearly straight line sloping downward — the famous **long tail**. Most of the probability mass sits in a tiny number of words; an enormous vocabulary of rare words occupies the tail.

---

## 3. Why Zipf Exists: Least Effort and the Efficiency–Robustness Trade-off

### Zipf's Cognitive Explanation: Least Effort

George Kingsley Zipf argued that the words we use most often **should cost us the least effort**. Frequent words should be:

- **Short** ("I", "a", "the" vs. "antidisestablishmentarianism")
- **Easy to produce** (common sounds, simple syllables)
- **Easy to retrieve** (highly activated in memory)

This is the **Principle of Least Effort**: speakers optimize for minimal cognitive and articulatory cost.

### The Information-Theoretic Explanation: Efficiency vs. Robustness

Language must simultaneously optimize two competing goals:

| Goal | What It Means | How Language Achieves It |
|------|--------------|--------------------------|
| **Efficiency** | Communicate with minimal effort/time | Short words, repeated regularities, redundancy removal |
| **Robustness** | Survive noise, distraction, errors | Longer words, redundant structure, recoverable messages |

**Efficiency** favors compression: if "the" is predictable, why encode it fully? **Robustness** favors redundancy: if a message is long and structured, you can reconstruct it even when bits are lost.

Language sits at the **Pareto-optimal balance** between these two pressures. That balance is what produces Zipf's Law.

### Why This Matters for NLP

A machine that learns human language patterns will **mimic human behavior** — producing text that "sounds right" because it follows the same statistical regularities. This is why language models trained on real text feel natural, and why models trained on random text do not.

---

## 4. The Monkey-Typing Experiment

### The Setup

Give a monkey a keyboard with 27 keys (26 letters + space). Let it press keys randomly. Generate a million characters. Split the output into "words" at spaces.

### The Surprising Result

Even this **random, nonsensical text follows a Zipf-like distribution**.

- A few short "words" appear very often (by chance)
- A long tail of rare "words" appears once or twice
- The number of possible words grows **exponentially** with word length

### What This Tells Us

Zipf's Law can emerge from **pure statistical sampling** — it does not require meaning, grammar, or human intention.

This is a crucial insight: **Zipf's Law is necessary but not sufficient for language**. Human language exhibits Zipf, but so does random noise. The difference is that human language has **structure** — predictable sequences, grammatical constraints, semantic coherence — that random text lacks.

### Try It Yourself

```python
import random

alphabet = 'abcdefghijklmnopqrstuvwxyz '
text = ''.join(random.choice(alphabet) for _ in range(1_000_000))
words = text.split()
# Count frequencies, sort by rank, plot log-log
# You will see a Zipf-like curve
```

---

## 5. Redundancy: Q Is Always Followed by U

### Shannon's Observation

Claude Shannon noticed a striking regularity in English: **wherever Q appears, U follows**. Think of:

- **q**uestion
- **q**uick
- **q**uack
- re**q**uest
- **q**uaint
- **q**uote

The U after Q is nearly 100% predictable. In information-theoretic terms, **the U carries almost no information** — you already knew it was coming.

### The Telegram Problem

Imagine sending a message by telegram, where you pay **per character**. The optimization goal is simple: **compress the message as much as possible without losing meaning**.

If U after Q is redundant, you can drop it: "qestion" is still readable. You save money.

This is the essence of **lossless compression**: remove predictable characters, keep the unpredictable ones.

### Other Redundancies in English

- Double letters: "o**cc**urrence", "su**cc**essful", "e**ng**ineering", "inte**ll**igence"
- Common bigrams: "th", "he", "in", "er", "an", "re"
- Common trigrams: "the", "and", "ing", "ion", "tio"

Each of these is a pattern your brain has learned to expect. When the pattern is violated, you notice — which is exactly how **typo detection** works.

### The Typo-Detection Insight

Have you ever written "recieve" and felt something was wrong before you consciously checked? That feeling comes from a **mismatch between the statistical pattern in your mind and what you're seeing**. Your internal bigram model says "ie" after "c" is rare; "receive" matches the pattern. This is your brain running an n-gram model in real time.

---

## 6. Language as a Probabilistic System

### The Core Hypothesis

Language is a **dynamic system whose future is shaped by its past history**. Each symbol (letter, word, phoneme) is not deterministic — it is **probabilistic**, conditioned on what came before.

\[
P(\text{next symbol} \mid \text{context})
\]

### Examples

- \(P(U \mid Q) \approx 1\) — U is almost certain after Q
- \(P(\text{"bank"} \mid \text{"fisherman went to the"}) > P(\text{"bank"} \mid \text{"I need a loan from the"})\)
- \(P(\text{"lamp"} \mid \text{"a small oblong reading"}) \gg P(\text{"lamp} \mid \text{"a"})\)

### The Key Insight

**Context reduces uncertainty.** The more context you have, the fewer choices remain, the more predictable the next symbol becomes. This is the fundamental intuition behind **all language modeling** — from n-grams to transformers.

---

## 7. Shannon's Game: N-Order Approximations

Shannon invented a game to model language probabilistically. The idea: start with random characters, then progressively add constraints from real English. At each order, the text looks **more like English**.

### The Ladder of Approximations

```
Order 0:  xkq zjmw pqlrb ftds ...
Order 1:  frib eotn aher spto ...
Order 2:  on at is the and ...
Order 3:  sum of the way in of the web ...
```

Let's walk up the ladder.

### 7.1 Zero-Order Approximation

**Rule:** Sample each character uniformly at random from 27 symbols (26 letters + space).

**Output:** `xkq zjmw pqlrb ftds ...`

**Properties:** No structure, no correlation, no predictability. This is the monkey-typing baseline.

**Entropy:** Maximum — each symbol is equally likely, so \(H = \log_2 27 \approx 4.75\) bits per character.

### 7.2 First-Order Approximation (Letter Frequencies)

**Rule:** Sample each character **independently**, but with the **frequency distribution of English**.

**The frequency plot:** On the x-axis, letters A–Z. On the y-axis, relative frequency. You'll see:

- **E** is most frequent (~12%)
- **T**, **A**, **O**, **I**, **N** are next
- **Z**, **Q**, **J**, **X** are rare (nearly touching the bottom)

**Output:** `frib eotn aher spto ...`

**Properties:** More English-looking than zero-order — E appears often, Z rarely — but still gibberish because **letter order is ignored**.

**Key point:** Each character is independent, but the **marginal probabilities** match English. This is a **unigram model**.

**Entropy:** Lower than zero-order, because the distribution is skewed — common letters carry less surprise.

### 7.3 Second-Order Approximation (Letter Bigrams)

**Rule:** Sample each character conditioned on the **previous character**, using English bigram frequencies.

**What this captures:**

- After Q, U is almost certain
- "th", "he", "in", "er" are common
- Double letters (cc, ll, ss, ee) are frequent
- "aa", "qz", "jz" are nearly impossible

**Output:** `on at is the and ...`

**Properties:** Real English words start appearing! "on", "at", "is", "the" are all valid. The text is still not fully English, but it's **recognizably language-like**.

**This is a bigram model.**

### 7.4 Third-Order Approximation (Letter Trigrams)

**Rule:** Condition each character on the **previous two characters**.

**Output:** `sum of the way in of the web ...`

**Properties:** Now we see **multi-word fragments** that look genuinely English: "sum of", "of the", "in of the". The text is approaching readability.

**This is a trigram model.**

### 7.5 Word-Level Approximations

The same ladder applies at the **word level**:

**First-order word model:** Sample words independently according to their frequency. Output: "the of and to a in..." — common words appear, but there's no grammar.

**Second-order word model (word bigrams):** Condition each word on the **previous word**. Pairs that never occur in English are **penalized**:

- "a and in" — should not occur
- "the and two" — should not occur
- "of the" — should occur
- "in the" — should occur

**Output:** `we head and in attack on ...` — meaningless, but **English-like**.

### 7.6 The Big Lesson

As you increase the order of approximation:

| Order | What It Captures | Output Quality |
|-------|-----------------|----------------|
| 0 | Nothing | Gibberish |
| 1 | Symbol frequencies | Gibberish with right letters |
| 2 | Symbol pairs | English-like fragments |
| 3 | Symbol triples | Recognizable English phrases |
| Word 1 | Word frequencies | Common words, no grammar |
| Word 2 | Word pairs | English-like but meaningless |

**Each constraint reduces uncertainty.** When you know what comes next, you have fewer choices. Fewer choices = lower entropy = more predictable = more English-like.

---

## 8. The Limits of N-Grams

### The Computational Complexity Problem

As you increase \(n\), two things happen:

1. **Entropy drops** (good — more predictable)
2. **Computational cost explodes** (bad — you need to store and query exponentially many n-grams)

At around **n = 7 or 8**, the approach **fumbles**: many n-grams never appear in your training data, so you have no frequency estimate for them. The model becomes sparse and unreliable.

### Long-Range Dependencies

N-grams also fail to capture **long-distance relationships**:

- "The key that the man who wore the red hat **was**..." — the verb agrees with "key", ten words back
- In Hindi: "Ram ne..." — the verb will be past tense, even if the sentence is a kilometer long

An 8-gram cannot see this far. Yet humans handle it effortlessly.

### Human Computational Limits

Why do humans use language the way they do? Because we operate under **three constraints**:

1. **Cognitive limit:** We can't store unlimited context in working memory
2. **Social limit:** We're under time pressure to communicate quickly
3. **Information access limit:** We don't have all relevant information available

Machines can handle more memory and more context, but the study of language modeling is fundamentally about **how humans generate and process language** under these constraints.

### The Takeaway

N-grams are a **useful first approximation** — they capture local patterns and reduce entropy. But they are not the whole story. Modern language models (transformers) solve the long-range dependency problem through **attention mechanisms**, but the core insight remains: **context reduces uncertainty**.

---

## 9. Entropy: Measuring Uncertainty

### The Intuition

We've established that language is probabilistic and that context reduces uncertainty. Now we need a **quantitative measure** of that uncertainty.

**Entropy** is that measure. It quantifies the **average surprise** of a message — or equivalently, the **average number of bits** needed to encode an outcome.

### The "A Small Oblong Reading Lamp" Demo

Consider the word sequence:

> "a ___"

How many possible next words are there? A monkey, a book, a watch, a small bus, a paper... **Many choices → high entropy.**

> "a small ___"

Fewer choices. **Lower entropy.**

> "a small oblong ___"

Most people don't know what "oblong" means. The choices shrink further. **Even lower entropy.**

> "a small oblong reading ___"

"Lamp" is the obvious answer. **Very low entropy.**

> "a small oblong reading lamp on the ___"

"Desk." Only a few people guess this — but it's the most likely answer. **Near-zero entropy.**

This is **context reducing uncertainty**, word by word. It's the same principle behind every language model.

### The Formula

Entropy \(H\) of a random variable \(X\) with outcomes \(x\) and probabilities \(p(x)\):

\[
H(X) = -\sum_{x} p(x) \log_2 p(x)
\]

Or equivalently:

\[
H(X) = \mathbb{E}[-\log_2 p(X)] = \mathbb{E}[\text{surprisal}]
\]

### What Each Part Means

- **\(p(x)\)** — the probability of outcome \(x\)
- **\(-\log_2 p(x)\)** — the **surprisal** of outcome \(x\) (rare events are surprising; certain events have zero surprisal)
- **\(\mathbb{E}[\cdot]\)** — the **expectation** (weighted average)
- **\(H(X)\)** — the **average surprisal** across all outcomes

### The Expectation Analogy

You already know the expectation formula:

\[
\mathbb{E}[X] = \sum_x x \cdot p(x)
\]

Entropy is the same structure, but with **surprisal** (\(-\log_2 p(x)\)) instead of \(x\):

\[
H(X) = \sum_x (-\log_2 p(x)) \cdot p(x)
\]

### Units

| Logarithm Base | Unit | Name |
|---------------|------|------|
| Base 2 (\(\log_2\)) | **bits** | Most common in information theory |
| Base 10 (\(\log_{10}\)) | **hartleys** | Named after Ralph Hartley |
| Base \(e\) (\(\ln\)) | **nats** | Used in some machine learning contexts |

### Simple Examples

**Fair coin:** \(P(H) = P(T) = 0.5\)

\[
H = -0.5 \log_2 0.5 - 0.5 \log_2 0.5 = 1 \text{ bit}
\]

**Biased coin:** \(P(H) = 0.99, P(T) = 0.01\)

\[
H = -0.99 \log_2 0.99 - 0.01 \log_2 0.01 \approx 0.08 \text{ bits}
\]

**Certain event:** \(P(H) = 1\)

\[
H = -1 \log_2 1 = 0 \text{ bits}
\]

**Uniform over 27 symbols** (zero-order approximation):

\[
H = \log_2 27 \approx 4.75 \text{ bits}
\]

### The Historical Note

Shannon introduced this formula in **1948** in "A Mathematical Theory of Communication". He was inspired by **Ralph Hartley's** earlier formula:

\[
n \cdot H = \log s
\]

where \(s\) is the number of possible messages and \(n\) is the message length. Hartley proposed this in the 1920s, but he didn't fully quantify what "information" meant. Shannon built on this foundation — **standing on the shoulders of giants**.

---

## 10. Redundancy in English: The 2.3-Bit Result

### The Binary Warm-Up

How many bits do you need to encode the numbers 1–8?

\[
2^3 = 8 \quad \Rightarrow \quad \text{3 bits}
\]

How many bits for 26 letters?

\[
2^5 = 32 \ge 26 \quad \Rightarrow \quad \text{5 bits}
\]

If English were a **random language** with no frequency patterns, you would need all **5 bits per character** to encode it.

### Shannon's Experiment

Shannon asked: **How much information does a character in English actually carry?**

He had human subjects guess the next character in English text, and measured how often they were correct. From this, he estimated:

> Reading a sequence of English text longer than **eight letters**, the actual information being transmitted drops to about **2.3 bits per character**.

### The Calculation

- **Raw encoding:** 5 bits per character (uniform over 26 letters)
- **Actual information:** ~2.3 bits per character
- **Redundancy:** \(5 - 2.3 = 2.7\) bits, or roughly **50%**

\[
\text{Redundancy} = 1 - \frac{H_{\text{actual}}}{H_{\text{max}}} = 1 - \frac{2.3}{5} \approx 0.54 \approx 50\%
\]

### The Compression Demo

Consider this sentence with vowels removed:

> "Th qck brwn fx jmps vr th lzy dg"

You can read it perfectly. The vowels were **redundant** — they carried less information than the consonants because they were more predictable from context.

This is **lossless compression**: the original message can be fully recovered from the compressed version.

### The Nokia Analogy

Remember texting on old Nokia phones? People wrote "hw r u" instead of "how are you" — more efficient, but now considered informal. The redundancy in full English ("how", "are", "you") makes the message robust but longer. Compression removes that robustness for efficiency.

### The Conclusion

English has **~50% built-in redundancy**. This redundancy is what makes the language **robust** — you can lose half the characters and still recover the message. But it also makes it **inefficient** — you need twice as many bits as the raw information content.

---

## 11. Entropy Is Not Sequence Randomness

### The Common Misconception

Many people think entropy measures **how random a sequence is**. This is **wrong**.

### What Entropy Actually Measures

Entropy measures the **uncertainty of the next symbol**, given the current state. It is:

- **Not** about the randomness of the entire sequence
- **Not** about how "disordered" a string looks
- **Yes** about the average surprisal of the next event

### A Revealing Example

Consider the sequence:

> "aaaaaaaaaaaaaaaaaaaa"

If you only look at symbol frequencies, the entropy of 'a' is 0 (it's certain). But if you look at the **sequence**, it's perfectly predictable — there's no randomness at all.

Now consider:

> "xkq zjmw pqlrb"

This looks random, but its per-symbol entropy might be high because each symbol is surprising.

**Entropy is about the next event, not the whole sequence.**

### The Formal Statement

\[
H(X) = \mathbb{E}[-\log_2 p(X)]
\]

This is the **average surprisal of a single outcome** — not a measure of sequence-level disorder.

### Why This Matters

When you see "entropy" in a machine learning paper, it usually means:

- **Cross-entropy loss:** the average surprisal of the true label under the model's distribution
- **Entropy regularization:** encouraging the model's output distribution to be uncertain (high entropy) or confident (low entropy)

In both cases, it's about the **next prediction**, not the sequence as a whole.

---

## 12. Efficiency vs. Robustness in Communication

### Shannon's Engineering Turn

Shannon's interest was not just language — it was **communication systems**:

- Transmitter → Channel → Receiver

He wanted to understand how to transmit messages **efficiently** and **reliably**, even over noisy channels. He got his insights from **language**, then applied them to **electrical communication** (flip-flops, zeros and ones).

### The Trade-off

| Strategy | Effect on Efficiency | Effect on Robustness |
|----------|---------------------|---------------------|
| **Remove redundancy** | ↑ Efficient (fewer bits) | ↓ Fragile (errors unrecoverable) |
| **Add redundancy** | ↓ Efficient (more bits) | ↑ Robust (errors recoverable) |

### Examples

**Remove redundancy:** "I love you" → "I lv u" — shorter, but if one character is lost, meaning may be ambiguous.

**Add redundancy:** "I love you" → "I love you, I love you, I love you" — longer, but even if one repetition is lost, the message survives.

### Shannon's Legacy

From these insights, Shannon developed:

- **The Mathematical Theory of Communication** (1948)
- **The Noisy Channel Coding Theorem** — the foundation of error-correcting codes
- **"Prediction in Printed English"** — a paper applying information theory to language

His work is the foundation of **all modern communication**: WiFi, 5G, satellite links, and the language models you use today.

---

## 13. Summary and Key Takeaways

### The Core Concepts

| Concept | Definition | Key Insight |
|---------|-----------|-------------|
| **Zipf's Law** | Frequency × rank ≈ constant | Few words are frequent; most are rare |
| **Least Effort** | Frequent words are short and easy | Language optimizes for cognitive cost |
| **Efficiency vs. Robustness** | Compression vs. error recovery | Language balances both |
| **Monkey Typing** | Random text also follows Zipf | Zipf is necessary but not sufficient for language |
| **Redundancy** | Predictable characters carry little info | Q→U, double letters, common bigrams |
| **N-Order Approximations** | Progressive constraints on language | Higher order = more English-like |
| **Entropy** | Average surprisal of next symbol | \(H = -\sum p \log p\) |
| **Surprisal** | \(-\log p(x)\) | Rare events are surprising |
| **Bits vs. Hartleys** | Log base 2 vs. log base 10 | Bits are standard in info theory |
| **50% Redundancy** | English carries ~2.3 bits/char vs. 5 raw | Half the bits are predictable |
| **Entropy ≠ Randomness** | Entropy is about next-event uncertainty | Not about sequence disorder |

### The One-Sentence Summary

> **Language is a probabilistic system with Zipf-distributed symbols and ~50% redundancy; Shannon's n-order approximations show that constraints reduce entropy toward English; entropy measures the average surprise of the next symbol (in bits), not the disorder of the whole sequence.**

### What's Next

This tutorial covers the **foundations** of information theory as applied to language. From here, you can explore:

- **Cross-entropy loss** — how language models are trained
- **KL divergence** — how to measure the difference between distributions
- **Mutual information** — how much one variable tells you about another
- **Perplexity** — the exponential of entropy, used to evaluate language models
- **Transformer attention** — how modern models capture long-range dependencies that n-grams miss

### Exercises

1. **Zipf in your data:** Download a public text corpus (Project Gutenberg, Wikipedia dump). Count word frequencies. Plot frequency vs. rank on a log-log scale. Does it follow Zipf's Law? What is the slope?

2. **Monkey typing:** Generate 1,000,000 random characters from a 27-symbol alphabet. Split into words. Count frequencies. Does the distribution look Zipf-like? How does it compare to real English?

3. **Entropy of a die:** Calculate the entropy of a fair 6-sided die. Then calculate the entropy of a biased die where 6 appears 50% of the time. Which is higher? Why?

4. **Compression:** Take a paragraph of English text. Remove all vowels. Can you still read it? Now remove all consonants. Can you still read it? What does this tell you about redundancy?

5. **N-gram generation:** Implement a bigram model trained on a small corpus. Generate 100 words. How English-like is the output? Try a trigram model. How does it compare?

6. **Typo detection:** Write a function that uses a bigram model to score words. Test it on common misspellings ("recieve", "teh", "adn"). Does it assign lower scores to the misspellings?

---

## Glossary

| Term | Definition |
|------|-----------|
| **Zipf's Law** | The observation that word frequency is inversely proportional to rank |
| **Long tail** | The large number of rare words that occupy the low-frequency end of the distribution |
| **Least Effort** | Zipf's principle that frequent words are short and easy to produce |
| **Redundancy** | Information that is predictable and therefore carries little surprisal |
| **Surprisal** | \(-\log p(x)\) — how surprising an outcome is |
| **Entropy** | The average surprisal of a random variable |
| **Bit** | Unit of information when using log base 2 |
| **Hartley** | Unit of information when using log base 10 |
| **N-gram** | A sequence of n symbols (letters or words) |
| **Bigram** | A 2-gram (pair of symbols) |
| **Trigram** | A 3-gram (triple of symbols) |
| **Cross-entropy** | The average surprisal of true labels under a model's distribution |
| **Perplexity** | \(2^{H}\) — the effective number of choices a model faces |

---

## Further Reading

- **Shannon, C. E. (1948).** "A Mathematical Theory of Communication." *Bell System Technical Journal.* — The founding paper of information theory.
- **Shannon, C. E. (1951).** "Prediction and Entropy of Printed English." *Bell System Technical Journal.* — The 2.3-bit result.
- **Zipf, G. K. (1949).** *Human Behavior and the Principle of Least Effort.* — The original Zipf's Law.
- **Cover, T. M., & Thomas, J. A. (2006).** *Elements of Information Theory.* — The standard textbook.
- **Jurafsky, D., & Martin, J. H.** *Speech and Language Processing.* — Chapter on n-gram language models.

---

*This tutorial is released under CC BY 4.0. Contributions welcome.*