# Quantifying Regularity in Language: From Turing and Shannon to Redundancy

## A Self-Contained Tutorial on Information, Uncertainty, and Compression

---

## Introduction: The Question That Started It All

Language is not random. If you have ever finished someone else's sentence, guessed the last word of a headline, or auto-completed a text message, you have already noticed what linguists and information theorists have known for decades: **language contains regular patterns**.

This raises a deeper question:

> **Can we quantify those regularities mathematically?**

This tutorial answers that question. By the end, you will be able to measure:

- **Predictability** — how guessable a message is
- **Uncertainty** — how many possibilities a message resolves
- **Information** — what actually changes when a message arrives
- **Redundancy** — what can be removed without losing meaning

These four concepts form the foundation of information theory, and they all trace back to a remarkable moment in history: two brilliant scientists, working at Bell Labs during World War II, who shared lunch and a single obsession.

---

## Part 1: What Is Redundancy?

Before we get to the history, let's define the central concept.

**Redundancy** is the part of a message that can be removed without losing meaning. It is the predictable padding that makes language robust.

Consider this sentence:

> "I woke up at 5 AM in the **morning**."

The word "morning" is redundant. "5 AM" already tells you it is morning. Delete it, and the meaning survives. That deletable padding is redundancy.

Here are three ways to recognize redundancy:

| Signal of Redundancy | Example |
|---|---|
| **Low information** | "ATM machine" — the "machine" adds nothing |
| **Repetition** | "I said it, I said it, I said it" |
| **High predictability** | "Keep the flower on the ___" → table |

**Key insight:** If you can predict what comes next, that next thing is redundant. Your brain's ability to auto-fill is direct evidence of measurable redundancy in language.

---

## Part 2: The Historical Setup — War Makes Information Expensive

Why does a tutorial about language begin in 1943?

Because war is where information became expensive. During World War II, powerful countries were fighting, and **classified information** was desperately sought by all sides. Defense ministries invested heavily in two technologies:

1. **Communication** — how to send messages securely
2. **Code-breaking** — how to intercept and decode enemy messages

The logic was simple and high-stakes:

- A country sends missiles and armies to other countries.
- Soldiers receive coded messages from a base station.
- Britain wanted to intercept those messages and decode them.
- Success meant saving lives or launching a counterattack.

Think of it as the world's highest-stakes game of hangman. Germany sends a scrambled message; Britain must guess the letters. Every correct guess saves lives. That pressure forced people to study messages **mathematically**.

---

## Part 3: Two Scientists, One Question

At Bell Labs in 1943, two remarkable scientists worked on communication and code-breaking:

### Alan Turing

- **Undergraduate:** King's College, Cambridge — where he independently proved the central limit theorem as an undergraduate
- **PhD:** Worked with supervisor Alonzo Church, formulating what became known as the Church–Turing thesis
- **Fascinations:** Chess, crossword puzzles, and typewriters
- **Key insight:** At heart, Turing was a **logic** person. His fascination with the mechanics of typewriters later led him to build a machine that broke German encryption — he figured out a way to decode the Enigma machine.

### Claude Shannon

- **Undergraduate:** Electrical Engineering and Mathematics at the University of Michigan
- **Master's:** MIT — his thesis was pioneering in digital circuit design theory, and later engineers built entire technologies on it
- **PhD:** MIT — worked on algebra for theoretical genetics (he went into biology)
- **Key insight:** Shannon was an **information** person. He wanted to understand what information actually is.

### Their Shared Obsession

Despite their different backgrounds, they shared one central question:

> **How can information and computation be represented properly?**

Turing approached it from the computation side: he wanted to understand computation theoretically and mathematically. Shannon approached it from the information side: he wanted to understand information itself.

**Lesson for learners:** These giants were students once, just like you. They solved problems, wrote papers, and had varied backgrounds. Reading their biographies reveals something important — **diverse backgrounds plus one deep shared question equals breakthroughs.** You do not have to pick one field forever.

---

## Part 4: The Famous Lunch Conversation

Turing and Shannon met regularly over lunch. They never discussed their classified projects, but they talked about technology, research, and ideas.

One day, Turing showed Shannon a paper about the **limits of computation** — what are the bounds of computation, what are the limitations, to what extent can you easily solve a problem. The paper was about the powers and limitations of an idealized computing machine. Turing was investigating a profound question:

> **How can a machine think?**

Shannon replied with a different vision: why not develop a machine that behaves more like a human, one that is culturally aligned? He wanted an **electronic brain**.

Turing disagreed. His response:

> "No, I'm not interested in developing a powerful brain. I am after just a mundane brain. Not a human brain — something that is very boring, something that you don't want to do as a human task. Just want to outsource it to someone else."

Turing wanted to **automate the boring**. He wanted machines to take over tasks humans would rather not do.

**Memory hook:** Shannon wanted a brain with a personality; Turing wanted a brain that does your homework. Both won — one became AI, the other became automation.

**The remarkable part:** It was 1943. They were talking about the future before transistors and electronic computers existed. Charles Babbage, the father of computers, would have been astonished to hear this conversation. That tells you something about the vision these people had.

**Their common foundation:** Both were ultimately about **formalizing information and computation**. And the underlying thing there? **Logic.**

---

## Part 5: Logic as the Bridge

### Turing's Side: Encoding Instructions as Numbers

Turing was encoding instructions as numbers. He converted instructions into decimal numbers, zeros and ones. He was an electrical engineer dealing with switches and relays, transferring codes using various switches.

The core idea: **all about symbols and mapping symbols to different compositions of numbers.** This is representation proper — mapping one set of objects into another.

**Simple example:** A light switch. OFF = 0, ON = 1. Your instruction "turn the light on" becomes the number 1. A whole recipe of instructions becomes a string of 0s and 1s. That is representation — translating words into numbers.

**A note on tensors:** In modern terms, a scalar is one number, a vector is a list, a matrix is a grid, and a tensor is the general container. Turing's 1943 idea — represent symbols as numbers — is the great-grandparent of the tensors used in modern AI.

### Shannon's Side: The Theory of Cryptography

Shannon was inspired by Turing's work and developed a mathematical theory of cryptography, which was later published.

The key insight Shannon took: **the signal that matters is the one that changes the probability of an event.**

---

## Part 6: What Is a Signal? The Probability-Changing Insight

This is the most subtle concept in information theory, and it deserves careful attention.

Ask yourself: **What is a signal?**

- Is it the probability of the event?
- Is it the thing that changes the probability of the event?

The answer: **both are signal, but the more important one is the thing that changes the probability.**

### The Class Cancellation Example

Imagine you know you have to go to class today. Your probability of attending is 1 — you are determined to go. Then you receive a message: "Class is cancelled due to a technical issue."

Your probability changes from 1 to 0. **That change is the information.**

Before the message: P(class happens) = 1 (certain)
After the message: P(class happens) = 0 (impossible)
The message **moved the probability**. That movement **is** the information.

**A message that changes nothing carries nothing.**

### The WWII Port Example

During the war, suppose Britain receives evidence about a location. One port could represent multiple possible fleet positions. There is uncertainty there — a probability distribution over possibilities. Turing's machine was measuring the probability of particular events in the code break.

When Britain receives some evidence, the uncertainty decreases. Things become more deterministic with the probability. The intercept **reduces uncertainty**, and how much it reduces is the information content.

**Memory hook:** Information is the nudge that moves your belief. No nudge, no news.

---

## Part 7: From Junk Data to Pattern

Shannon's next move was profound: he decided to study **language itself**.

Here is the reasoning:

- Data is just a signal. Without insight or structure, it is useless.
- You may have data that looks like junk because you do not understand the signal.
- Your goal is to find a real pattern in the data.
- How do you find patterns? You look at what changes the data, what changes the signal, what structures are involved.

Why language, of all things?

- At the time, the most easily available data was language itself.
- Shannon spoke it and understood it well.
- **The best dataset is often the one you already understand.**

Two properties make language measurable:

1. **The message is tangible** — if there is a pattern in language, the message can be calculated and measured.
2. **Language is countable** — language samples are discrete; you can count them.

These two properties also exist in real communication, and they motivated Shannon to find patterns in language.

---

## Part 8: Redundancy, Compression, and the Telegram

Shannon focused on **redundancy** — what kind of information gets duplicated, or how much you have to say.

The framing: assume we live in a noisy world where everything is expensive. How much do you have to say effectively without huge expense? What language do you use?

### The Telegram Example

Consider the old Telegram (not the app — the actual telegram service used to send fast messages, like a fax). It was expensive: **each word and each letter cost money.** So you wanted to make your message as compressed and meaningful as possible.

This creates a trade-off: **efficiency versus robustness.**

- Too much redundancy: expensive and slow.
- Too little redundancy: fragile and easily misunderstood in noise.

### The Compression Chain

Here is the logical flow Shannon formalized:

```
Noisy world + expensive messages
        ↓
How much can we compress without losing meaning?
        ↓
To compress, you must know the REDUNDANCY
        ↓
Once compressed → that is the CORE MESSAGE
        ↓
Receiver decodes it back
```

**In Shannon's words:** Once you know the redundancy, then only can you compress. Once you have compressed, that is the core message you want to send. Later, it will be decoded. This redundancy helps us in understanding and recovery.

**Memory hook:** Redundancy is the backup copy; compression is the shipping label; the core message is the parcel.

---

## Part 9: Language Is Predictable — The Fill-the-Blank Test

The final piece: **language is not random. It is predictable. That is why there is redundancy.**

Try this interactive demonstration. Complete each sentence:

- "Keep the flower on the ___!" → **table**
- "I'm going to the ___ to buy milk." → **store**
- "Twinkle twinkle little ___!" → **star**

Every blank you filled without thinking is a **measured redundancy**. Your brain did statistics. Shannon simply wrote down the math your brain was already doing.

**The rule:** If you can predict the blank, it is redundant.

---

## Part 10: The Zip File Connection

A natural question: is the ZIP algorithm (WinZip, WinRAR) based on these ideas?

The answer: it is not directly inspired by Shannon, but **all these things converge.** They share interrelated ideas. Redundancy is also related to the **principle of least effort** — nature and engineers both take the shortest path.

When you zip a file, the computer finds repeated patterns and replaces them with short codes. That is redundancy-hunting in action. ZIP does not cite Shannon in its code, but it obeys his law.

---

## Summary Table

| Concept | One-Line Memory |
|---|---|
| The big question | Can we *measure* language's regularity? |
| Redundancy | The deletable padding that still keeps meaning |
| WWII context | War made information expensive, forcing measurement |
| Turing vs. Shannon | Computation lens vs. information lens; both rooted in **logic** |
| Turing's lunch view | Automate the boring ("mundane brain") |
| Shannon's lunch view | Build a human-like "electronic brain" |
| Signal | Anything that **changes the probability** of an event |
| Class-cancelled example | P goes 1 → 0; the shift *is* the information |
| Why language? | Tangible, countable, and already understood |
| Telegram | Compress → core message → decode |
| Fill-the-blank | If you can predict it, it is redundant |
| ZIP | Converges with the same idea |

---

## The One-Liner to Remember Forever

> **Language is predictable. Predictability is redundancy. Redundancy is compressible. And information is the surprise that survives compression.**

---

## Further Exploration

If you want to go deeper, here are the threads to pull:

1. **Alan Turing** — Read about his undergraduate proof of the central limit theorem, his PhD with Church, and his work on the Enigma machine.
2. **Claude Shannon** — Read his master's thesis on digital circuit design and his later work on cryptography and information theory.
3. **The Imitation Game** — Watch the film about Turing and Bletchley Park; it brings the WWII code-breaking context to life.
4. **Shannon's approach to language** — The next step is studying how Shannon actually measured redundancy in English text.

---

## Exercises

1. **Redundancy hunt:** Find three sentences in a news article where a word or phrase could be deleted without changing meaning. What did you delete?
2. **Probability shift:** Write a message you received today. What probability did it change? If it changed nothing, was it information?
3. **Fill the blank:** Create five sentences with blanks. Test a friend. Which blanks were easiest to fill? What does that tell you about redundancy?
4. **Compression trade-off:** Write a 20-word message. Then compress it to 10 words without losing meaning. Then to 5. At what point does robustness break?

---

*This tutorial is released as open-source educational material. Share it, remix it, and build on it.*