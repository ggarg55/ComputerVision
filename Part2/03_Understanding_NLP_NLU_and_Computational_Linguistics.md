# Understanding NLP, NLU, and Computational Linguistics: A Complete Tutorial

> **Learning Objectives**
> By the end of this tutorial, you will be able to:
> - Distinguish between NLP, NLU, and Computational Linguistics
> - Explain what historical systems like ELIZA and SHRDLU teach us about "understanding"
> - Describe the recipe for building an NLU system and how to evaluate it
> - Recognize the different types of ambiguity that challenge language systems
> - Understand the statistical foundations of modern language models
> - Articulate why "computation without linguistics" is a problem
> - Apply the NLP Trinity framework to reason about real-world problems

---

## 1. Three Fields, One Goal

When people say "AI understands language," they often blur together three distinct disciplines. Let's separate them with a simple analogy.

Imagine three people looking at the same car:

- **The mechanic** cares about how the engine works so they can *build and fix* things. This is **Natural Language Processing (NLP)** — application-focused and engineering-driven.
- **The physicist** cares about the *laws* behind why the car moves. This is **Computational Linguistics (CL)** — theory-focused and science-driven.
- **The driver** cares about *actually getting somewhere meaningfully*, using both the engineering and the science. This is **Natural Language Understanding (NLU)**.

### 1.1 Natural Language Processing (NLP)

NLP focuses on the **engineering side**, with specific applications in mind. The goal is to build systems that solve practical problems:

- **Information retrieval** — searching text and extracting insights from unstructured data
- **Speech systems** — text-to-speech and speech-to-text conversion
- **Sentiment and emotion analysis** — detecting what a piece of text, speech, or video conveys

The key idea: language is treated as **data** (information), and you build applications on top of it.

### 1.2 Computational Linguistics (CL)

CL asks a different question: *how can we study linguistic phenomena using computational methods?*

The goal here is **scientific**. You start with linguistic theories — about grammar, structure, and the nature of language — and build computational models inspired by those theories. The emphasis is on understanding *why* language works the way it does, often with cognitive constraints in mind.

### 1.3 Natural Language Understanding (NLU)

NLU is the most ambitious of the three. It **incorporates both NLP and CL** with a shared goal: develop a system that *meaningfully represents* natural language and *understands* it.

### 1.4 The Shared Pipeline

All three fields follow the same iterative loop:

```
Linguistic descriptions (from data)
        ↓
Representations of the data
        ↓
Build a computational model
        ↓
Test & validate (empirical evaluation)
        ↓
Identify failures → improve → repeat
```

Modeling alone is never enough. You must **validate** — find the scenarios where your model fails and the circumstances where it succeeds — and iterate until the model genuinely captures human language.

---

## 2. Three Historical Systems That Teach Us What "Understanding" Means

### 2.1 ELIZA (1966) — The Illusion of Understanding

Developed at MIT by Joseph Weizenbaum, **ELIZA** was an early chatbot designed to act as a psychotherapist.

Here's a representative exchange (ELIZA's responses in caps):

> **Human:** Men are all alike.
> **ELIZA:** IN WHAT WAY?
> **Human:** They're always bugging us about something or other.
> **ELIZA:** CAN YOU THINK OF A SPECIFIC EXAMPLE?
> **Human:** Well, my boyfriend made me come here.
> **ELIZA:** YOUR BOYFRIEND MADE YOU COME HERE.
> **Human:** I'm depressed much of the time.
> **ELIZA:** I'M SORRY TO HEAR THAT YOU ARE DEPRESSED.

**What's the pattern?** ELIZA picks up keywords from the user's message, applies simple rules, and reflects them back. Notice "depressed" appearing in both the input and the response. This is **lexical repetition** combined with **pattern matching** — not linguistic analysis.

**A memorable analogy:** Imagine a parrot that repeats the last noun you said. You say "I lost my keys," it says "Why did you lose your keys?" You feel heard — but the parrot has no idea what a key is.

**Takeaway:** Producing an appropriately-looking response is **not the same as understanding**. This insight motivated the next generation of systems.

### 2.2 SHRDLU (Terry Winograd, MIT) — Language Meets the World

SHRDLU operated in a **micro-world** of colored blocks. When told "Pick up a big red block," it would actually pick one up.

The interaction looks like this:

> **Human:** Pick up a big red block.
> **SHRDLU:** OK.

The "OK" is trivial — the *real* proof of understanding is the action itself. SHRDLU had to:

1. **Parse the language** — identify the action ("pick up") and the object ("big red block")
2. **Build a representation of the world** — which blocks exist, what colors and sizes they are
3. **Reason** — compare objects to find the one that is both *big* and *red*
4. **Act** — use robotic limbs to grasp the correct block
5. **Respond** — acknowledge completion

**The pipeline, compared side by side:**

```
ELIZA:   Language → Pattern Matching → Response

SHRDLU:  Language → World Representation → Reasoning → Action → Response
```

SHRDLU also possessed two kinds of knowledge — a distinction worth memorizing:

- **Declarative knowledge** — *facts* (what a block is, what red means)
- **Procedural knowledge** — *skills* (how to pick something up, with what force)

**Limitations:** SHRDLU was confined to a single micro-world, worked only in English, and had no multimodal capability.

**Takeaway:** If language is understood as a body of knowledge, SHRDLU showed us how that knowledge could be **represented** in a cognitive system and **integrated** with general knowledge about the environment.

### 2.3 The Turing Test and the Limits of Imitation

The Turing Test asks: if a machine can fool a human interrogator into thinking it's human, have we achieved AI?

Three important challenges to this idea:

**Challenge 1 — Mimicry is not thought.** A machine can produce human-like responses without thinking. Impressive output does not prove genuine understanding.

**Challenge 2 — The goal may be to exceed humans, not just match them.** In domains like robotic precision, AI can achieve what humans cannot. The goal is not only to compete with human minds but to go beyond them.

**Challenge 3 — The Total Turing Test.** Real intelligence requires **interacting with the environment** — having senses, perceiving the world, and acting within it. This is where SHRDLU pointed.

**The deepest challenge — qualia and the explanatory gap:** You can measure the wavelength of red light (700 nm) perfectly. But no measurement tells you *what red feels like*. A machine can label "red" flawlessly and never experience redness. That subjective, experiential gap is called **qualia** — and modeling it remains an open problem for AI.

---

## 3. How to Build an NLU System: The Recipe

Here is the three-step recipe for developing any language understanding system.

### Step 1: Get a Dataset (Corpus)

The data must **represent language** — and it need not be text alone. It can be **multimodal**: text, audio, and video.

Data must be **annotated**. For example, given the sentence "John sat near the bank," you might tag:

- **Part of speech** for each word (noun, pronoun, adverb, adjective)
- **Root forms** — e.g., "running" → "run"

### Step 2: Build a Computational Model

Three families of models:

- **Machine learning models** — learned from examples
- **Distributional models** — capture frequencies and regularities in the data
- **Rule-based models** — grammar rules extracted from the data itself

### Step 3: Test and Validate Against Humans

A model that emits responses is not done. Those responses must be **validated against human performance**:

- Responses should look appropriate, the way a human would respond
- Decisions the machine makes should reflect the decisions humans make

### 3.1 Evaluation Metrics

**Standard metrics** (e.g., for a binary classifier):

- **Accuracy**
- **Confusion matrix**
- **F1 score**

**Generation metrics:** **BLEU score** and similar measures evaluate generated text against references.

**Cognitive metrics** (inspired by psycholinguistics):

**Surprisal.** Natural human language has *low* surprisal. The probability of a sentence a machine generates should be high; equivalently, its surprisal should be low.

*Simple example:* "Apples and ___" → "oranges" feels natural (low surprisal); "Apples and coconut" feels odd (high surprisal). Human language tends toward the natural.

**Dependency length.** According to **Dependency Locality Theory**, human sentences have *lower* dependency length than chance. If you randomly permute the words of a sentence, related words scatter far apart; in natural sentences, dependents stay close.

*Simple example:* "The cat that chased the mouse ran" keeps related words together. Shuffle it randomly and the links stretch across the sentence.

### 3.2 The Hypothesis-Testing Loop

The full workflow:

```
Identify corpus → State hypothesis (e.g., "human sentences have low surprisal")
    → Compute metrics from the corpus → Build model
    → Validate: do machine-generated sentences match human metrics?
```

---

## 4. The Ambiguity Problem

Ambiguity is one of the central challenges of language understanding. And it is not a machine-only problem — *humans get confused too*. When you say "John sat near the bank," no one knows which bank you mean. If we get confused, imagine the poor machines.

### 4.1 The Star Example

Consider this sentence:

> **"A computer that understands you like your mother."**

It has (at least) **three distinct meanings**:

1. The computer understands you **as well as your mother understands you**.
2. The computer understands **that you like your mother**.
3. The computer understands you **as well as it understands your mother**. (The funniest one.)

### 4.2 The Speech Recognition Twist

Now imagine this sentence was *spoken* and transcribed by an ASR system. The phrase "you like your mother" can be transcribed as "**you're** like your mother" — phonetically nearly identical, but a completely different meaning. This is a serious challenge for speech systems.

### 4.3 The Domain Twist

Words carry domain-specific meanings. Is "mother" used in vinegar-making? If you work in that trade, the sentence means something entirely different. Domain knowledge can break a system that seems to work elsewhere.

### 4.4 The Classic Failure Story

Ask a current flagship model to "make a cake out of a pencil," and it may cheerfully attempt it — producing a cake made of pencil. Even today's most powerful models fail on such classic ambiguity problems.

### 4.5 Two Types of Ambiguity

**Lexical ambiguity** — a word has multiple meanings.

> "John sat near the **bank**." (river bank? money bank?)

The ambiguous word changes the meaning of the whole sentence.

**Structural ambiguity** — ambiguity at the level of phrase or constituent structure.

> "John saw Mary **with the telescope**."

Who has the telescope — John or Mary? The phrase "with the telescope" can attach to either. You can build two different sentence structures with two different meanings.

*Bonus example:* "Copy the local patient files to the disk." Is "local" modifying *patient* or *files*? Genuinely unclear — and consequential.

---

## 5. The Statistical Revolution

### 5.1 "You Shall Know a Word by the Company It Keeps"

This old saying captures a foundational idea: a word's meaning can be inferred from the words that co-occur with it. Co-occurrences in language data are a rich source of information.

*Simple example:* You never need a dictionary to guess that "bank" near *money, deposit, loan* means a financial institution, while "bank" near *river, water, fish* means a riverbank. **The neighbors reveal the meaning.**

This raises a question: **Is NLP fundamentally a massive knowledge-engineering project — or a data-driven one?** Perhaps feeding more data yields better co-occurrences and better numbers.

### 5.2 Shannon and Information Theory

Claude Shannon, the father of information theory, took a radical stance: he was **not interested in what a message means**. He was interested in **how surprising or uncertain** a message is given prior context.

The information content of a word depends on its **probability**, not its meaning.

**The fill-in-the-blank demo:** "Apples and ___" → almost everyone says "oranges." Why? The naturalness reflects a **conditional probability**: P(orange | apples and).

**The rule to memorize:**

> **Higher probability → lower information. Lower probability → higher information.**

Surprising events teach you something; expected events don't. This insight is the foundation of **next-word prediction** and is used every day in large language models.

---

## 6. Computation Without Linguistics?

Modern large language models are driven by frequencies and next-word prediction. This raises a sharp question: **computational linguistics without linguistics — where is the language?**

The "C" (computation) is clearly present. The "L" (linguistics) needs to be brought back. Here is the evidence that frequency-only models fall short.

### 6.1 Negation

> "I will **not** go to the school."

But add a comma:

> "I will not, go to the school." → "Definitely I'll go!"

Same words, opposite meaning. So can we say "not" always negates? No. Models struggle with this systematically — and it gets worse across languages and with double negations.

### 6.2 Events

Models handle poorly the various events embedded in sentences — who did what to whom, when, and why.

### 6.3 Coreference Resolution

> "Ram is a good boy and then **he** went to school."

"**He**" refers to Ram. Transformer models are notoriously weak at this kind of **coreference resolution** — tracking which expression refers to which entity across a text.

### 6.4 Rare Constructions

Rare constructions are rare precisely because they don't appear often in training data. Their signal will not be learned by a frequency-based model. Humans handle rarity gracefully; models cannot. **How to learn from rare but informative data** is an open challenge.

### 6.5 Low-Resource Languages

Consider the Munda language spoken in Jharkhand, India. It has little digital presence. Will AI work well for it? No. Even Hindi is low-resource in a crucial sense: plenty of sentences exist online, but they lack **linguistic annotation** — the structured data needed to study how the language actually works.

The conclusion: we need linguistically rich, annotated data so models can be trained to genuinely understand language.

---

## 7. The NLP Trinity

The challenges of language understanding form a **three-axis problem**:

```
                     LANGUAGES
                    (Hindi, Marathi,
                     English, Munda...)
                         /\
                        /  \
                       /    \
                      /      \
              TASKS ---------- ALGORITHMS
        (morphology, POS,      (HMM, RNN,
         chunking, parsing)     transformers...)
```

**Axis 1 — Languages.** Each language carries its own complexity and language-specific properties.

**Axis 2 — Processing tasks** (increasing complexity):

- **Morphology** — the study of words
- **POS tagging** — label each word as noun, verb, adjective, etc.
- **Chunking** — group words into phrases. "With a telescope" is one phrase; "Mary with" is not.
- **Parsing** — determine which word links to which word

**Axis 3 — Algorithms.** HMMs, RNNs, transformers, and countless others.

**The degree-of-freedom problem:** Fix one model (say, an HMM) and it may work for one task in one language — but fail elsewhere. A model that succeeds on English parsing may fail on Marathi morphology. There is enormous freedom on all three axes simultaneously.

This is why the field is pursuing **universal, multilingual models** — and why the field is so wide: **any tiny problem you work on sits inside this vast three-dimensional space.**

---

## 8. Key Takeaways

1. **Three lenses:** NLP (engineering and applications), CL (science and theory-driven models), NLU (meaningful representation, combining both).
2. **Three historical lessons:** ELIZA proves surface pattern matching is not understanding. SHRDLU grounds language in world representation, reasoning, and action. Turing and qualia remind us that mimicking is not experiencing.
3. **The recipe:** Corpus (annotated, possibly multimodal) → computational model (ML, distributional, or rule-based) → validate against humans using metrics such as accuracy, F1, BLEU, surprisal, and dependency length.
4. **Two statistical pillars:** Co-occurrence ("know a word by the company it keeps") and Shannon's information theory (probability, not meaning; low surprisal feels natural).
5. **The central failure mode:** Ambiguity — lexical (bank) and structural (telescope), plus ASR homophones ("you like" vs. "you're like") and domain-specific meanings.
6. **The open debate:** Computation without linguistics struggles with negation, coreference, events, rare constructions, and low-resource languages.
7. **The Trinity:** Languages × Tasks × Algorithms. No single model wins everywhere.

## 9. Questions to Keep Thinking About

- Are large language models **cognitive models**? They predict the next word — does that constitute understanding?
- If a machine merely predicts the next word well, is that enough — or do we need parsers and dependency graphs that encode linguistic structure?
- What would it take to model **qualia** — subjective experience — in an AI system?
- How can we learn from **rare but informative** constructions that frequency-based models miss?

---

## 10. Further Reading

The ideas in this tutorial draw on foundational work in NLP, computational linguistics, and cognitive science. Topics to explore next:

- Tokenization and its challenges
- Morphology, POS tagging, chunking, and parsing
- Dependency grammar and dependency locality theory
- Information theory and surprisal in language
- Critiques of frequency-based language models (Bender & Koller and related work on meaning, form, and understanding)

---

*This tutorial is released under open-source terms. Contributions and corrections are welcome.*