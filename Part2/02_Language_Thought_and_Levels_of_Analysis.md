# Foundations of Vision & Language
## Tutorial 2: Language, Thought, and the Three Levels of Analysis

---

## Introduction

This tutorial explores what it means for a machine to understand language. We begin with a linguistic case study that challenges universal assumptions about grammar, move through the philosophy of mind and computation, and arrive at a practical framework that engineers and scientists use to design intelligent systems.

By the end, you will be able to:

- Explain why recursion matters in linguistics using the Pirahã language as a case study
- Describe how machines "understand" language through vocabulary mapping
- Apply David Marr's three levels of analysis to any information-processing problem
- Distinguish between the classical Turing Test and the Total Turing Test
- Identify NLP systems in everyday objects

**Prerequisites:** Basic familiarity with what language is, and an open mind about how minds and machines process information.

---

## Part 1: The Pirahã Language — When Grammar Meets Culture

### 1.1 The Universal Grammar Assumption

Modern linguists have long believed that every human language shares certain **genetic transformations** — universal symbolic structures embedded in the brain. The idea is that all languages, despite surface differences, share deep structural commonalities.

This assumption was challenged by research conducted at MIT starting around 2011, when researchers built a database of the **Pirahã language** — a language spoken by an Amazonian tribe — to test whether any language could lack a supposedly universal feature.

### 1.2 What Is Recursion?

Before understanding the Pirahã case, you need a working definition of **recursion** in language.

Consider a main sentence, called a **matrix clause**. Inside that matrix clause, you can embed another clause — an **embedded clause**. And inside that embedded clause, you can embed yet another. The process can continue indefinitely.

**Example:**

> John said [that Mary had water]

Now embed further:

> John said [that Mary said [that Raj had water]]

And further still:

> John said [that Mary said [that Raj said [that Priya had water]]]

Each bracket represents a nesting level. You can keep opening new brackets and close them all later. This nesting capacity is **recursion** — the ability of a grammatical structure to contain another instance of itself.

**Analogy:** Think of Russian nesting dolls. Open one doll, find a smaller doll inside. Open that one, find an even smaller one. You can imagine infinitely many dolls nested within each other. Recursion in language works the same way — each clause can contain another clause, which can contain another, without any principled limit.

### 1.3 The Pirahã Finding

After feeding data into their database — a process that took roughly three months — the MIT researchers concluded that the Pirahã language **lacks recursion**. The embedded-clause structure that seems natural in English, Hindi, and most documented languages simply does not appear in Pirahã.

This is a profound finding. If recursion were truly universal — hardwired into the human brain — then no language should lack it. The Pirahã case suggests that at least some aspects of grammar are shaped by culture and environment rather than biology alone.

### 1.4 Culture and Groundedness

The Pirahã people's language reflects their worldview. An elder named Koho explained:

- They do not believe in God — they consider the concept foreign.
- They do not want to learn about things "high above the sky."
- They want to remain **grounded** — close to the earth.

Their language mirrors this groundedness. It lacks creation myths, lacks numbers beyond small quantities, and — crucially — lacks the nested, abstract embedding that recursion enables. Language, in other words, is not separate from thought and culture. It grows out of them.

**Key takeaway:** Language is not a free-floating formal system. It is shaped by the culture and cognitive habits of its speakers. Any machine that hopes to understand language must grapple with this coupling.

---

## Part 2: How Machines "Understand" Language

### 2.1 Vocabulary as a Lookup Table

A computer does not understand meaning. It understands only **0s and 1s**. So how does a machine process a sentence like "Mary had water"?

The answer is **vocabulary mapping**. Every word in the machine's vocabulary is assigned a number:

| Word | Assigned Number |
|------|-----------------|
| water | 1 |
| Mary | 20 |
| had | 21 |

When the machine encounters "Mary had water," it converts the sentence into the sequence [20, 21, 1] and searches its vocabulary to confirm those numbers exist. That is the entirety of "understanding" at this level — a lookup operation.

**Analogy:** Imagine a librarian who cannot read any book but has a perfect index-card catalog. When you ask for "Mary had water," the librarian checks whether cards #20, #21, and #1 exist in the catalog. If they do, the request is valid. The librarian has "processed" the sentence without understanding a single word.

### 2.2 Repetition and Learning

Human language is fundamentally repetitive. The word "water" might be spoken thousands of times because everyone drinks water. The machine does not need to relearn "water" each time — it simply recurses to the existing vocabulary entry and reuses it.

This is why **vocabulary size matters**. If the machine's vocabulary contains the words you use, it can process your sentence. If not — say you speak Latin or French and the machine only knows English — communication fails.

### 2.3 The Child Metaphor

The most important design principle for language systems is this: **treat the computer as a child**.

A child acquires language gradually — first words, then combinations, then complex structures. The child does not arrive with a fully formed grammar; they build it through exposure and repetition. Similarly, a language model must be trained incrementally, with the understanding that it is learning patterns rather than being programmed with rules.

This metaphor will recur throughout the course. When you design a language system, ask yourself: *How would a child learn this?* The answer often reveals the right architecture.

---

## Part 3: Two Lenses — Science and Engineering

When you think about **Natural Language Understanding (NLU)** — or any information-processing problem — you must hold two perspectives simultaneously.

| Perspective | What It Provides | Failure Mode If Missing |
|-------------|------------------|-------------------------|
| **Engineering** | A system that works | You have a black box — it works, but you don't know why |
| **Science** | An explanation of how it works | You have theory but no working system |

**Analogy:** An engineer who only builds is like a chef who can cook a dish but cannot explain the recipe to anyone else. A scientist who only theorizes is like a food critic who can describe a dish but cannot cook it. The most valuable contributor is the one who can do both — build the system *and* explain its workings.

Throughout this course, you will be expected to think from both perspectives. A system that works but is unexplainable is fragile. A theory that never produces a working system is sterile. The goal is integration.

---

## Part 4: The Chain of Questions — Can Machines Think?

The central philosophical question of this tutorial unfolds as a ladder. Each rung rests on the one below it.

### 4.1 The Ladder

**Rung 1: Can a machine understand language?**

This is the surface question. But language is not an isolated phenomenon.

**Rung 2: Language is linked to thoughts, and thoughts are linked to thinking. So — can a machine think?**

If understanding language requires thought, then the language question reduces to the thinking question.

**Rung 3: The mind is an information-processing system.**

The mind feeds on information. It receives signals through **perception** — the five perceptual systems: **eye, ear, mouth, smell, and touch**. These are the input channels. The mind processes these signals into thoughts, decisions, and utterances.

**Rung 4: Do machines carry out information processing just like the mind?**

If the mind is an information-processing system, and machines are information-processing systems, then perhaps the same framework applies to both.

**Rung 5: Humans have cognitive processes and cognitive faculties — but with limits.**

Human cognition operates under constraints:

- **Limited working memory** — we can only hold so much in mind at once.
- **Limited information** — we never have complete knowledge.
- **Limited time** — we must utter sentences and make decisions quickly.

Despite these limits, humans make rapid, effective decisions through cognitive mechanisms that are largely automatic.

**Rung 6: Does the machine follow those same cognitive processes?**

If yes, then we can ask the engineering question: **Can we develop a computational system that mimics those cognitive processes?**

**Rung 7: Can we reverse-engineer the cognitive processes — such as language and perception?**

This is the ultimate goal. If we can understand how the mind processes language and perception, we can build machines that replicate those processes.

### 4.2 The Kitchen Analogy

Think of your mind as a kitchen:

- **Ingredients** enter through five doors — your senses.
- The kitchen has **limited counter space** — your working memory.
- There is a **dinner rush** — time pressure forces fast decisions.
- The kitchen produces **dishes** — thoughts, sentences, actions.

The question is: can we build a **robot kitchen** that not only produces the same dishes (engineering) but uses the same recipes and the same limited counter space (science)?

### 4.3 Course Alignment

This course, Foundations of Vision & Language, sits directly on this ladder:

- **Vision** = the perception part
- **Language** = the language part

Cognitive computer vision, natural language understanding, and AI all share the same endeavor: understanding and replicating how minds process information. This is the bigger goal that motivates everything in this course.

---

## Part 5: David Marr's Three Levels of Analysis

### 5.1 The Framework

Any information-processing system — whether it processes vision, language, or any other signal — can be understood at three distinct levels. This framework was proposed by **David Marr** in the 1970s and remains foundational.

| Level | Question It Answers | Focus |
|-------|---------------------|-------|
| **1. Computational** | **WHAT** is the goal, and **WHY** do we want it? | The problem statement and its motivation |
| **2. Algorithmic (Representational)** | **HOW** — what representation for input and output, and what algorithm transforms input into output? | The rules and procedures |
| **3. Implementational** | **WHERE** — on what physical hardware or brain structure is the algorithm realized? | The physical substrate |

### 5.2 Why This Framework Suits Computer Scientists

Marr's framework was motivated by the advent of computers in the 1950s. Computer scientists naturally think in these terms:

- You have a **computational problem** (Level 1).
- You solve it with an **algorithm** (Level 2).
- The algorithm **runs on hardware** — a CPU (Level 3).

When it comes to the mind, the brain is the **implementational layer**. The question is: where exactly in the brain does a given algorithmic step — a decision, a perception — take place?

### 5.3 Worked Example: Making Tea

| Level | Tea-Making Example |
|-------|-------------------|
| **Computational** | "I want tea because I am sleepy." (Goal + motivation) |
| **Algorithmic** | Boil water → add tea leaves → steep → strain → pour. (Input: water and leaves. Output: tea.) |
| **Implementational** | Kettle, stove, cup, teapot. (The physical objects that execute the algorithm.) |

### 5.4 Worked Example: Recognizing a Cat in an Image

| Level | Cat-Recognition Example |
|-------|------------------------|
| **Computational** | "Determine whether an image contains a cat." (Goal) |
| **Algorithmic** | Extract features (edges, textures, shapes) → compare against a representation of "cat" → output yes/no. |
| **Implementational** | Run the algorithm on a GPU or in the visual cortex of a brain. |

### 5.5 Worked Example: Parsing a Sentence

**What is parsing?** Parsing means breaking a sentence into its constituent pieces and joining them back together so that you understand what the sentence means. It is the process of analyzing a sentence "in detail" so that its structure reveals its meaning.

| Level | Parsing Example |
|-------|----------------|
| **Computational** | "Determine the grammatical structure and meaning of a sentence." |
| **Algorithmic** | Tokenize → tag parts of speech → build a parse tree → extract semantic relations. |
| **Implementational** | Run the parser on a CPU, or map it to language-processing regions of the brain. |

**Analogy for parsing:** Parsing is like disassembling a Lego castle into individual bricks, labeling each brick, and then rebuilding the castle while knowing exactly which brick goes where. The rebuilt understanding — the labeled structure — is the meaning.

---

## Part 6: The Robot-Flying Exercise

### 6.1 The Task

Apply Marr's three levels to a concrete engineering problem: **make a robot fly**.

This task was chosen deliberately over software-only tasks (like cat-recognition or parsing) because it forces all three levels to come together:

- A **software engineer** typically works at Levels 1 and 2 only.
- A **hardware engineer** must work at all three levels, with Level 3 being most critical.
- Making a robot fly requires **software and hardware together** — the full stack.

**Analogy:** Cat-recognition is a recipe on paper. Robot-flying is cooking in a real kitchen with real heat — you cannot skip the physical layer.

### 6.2 Common Mistake: Jumping to Method

When presented with a problem, engineers often rush directly to the algorithmic or implementational level. They start listing motors, sensors, and control algorithms before they have clearly defined the goal.

**The correction:** First understand the problem at the **computational level**. Define WHAT you want and WHY. Only then should you proceed to HOW and WHERE.

### 6.3 A Complete Three-Level Analysis

**Level 1 — Computational (WHAT and WHY):**

> Make the robot fly.

Why? Various motivations might drive this — delivery systems, surveillance, search and rescue, or simply the challenge of flight. The motivation determines the constraints and priorities.

**Level 2 — Algorithmic (HOW):**

To fly, the robot must generate **thrust** greater than gravity. If mimicking a bird:

1. **Flap the wings** to push air downward.
2. The downward push generates **upward thrust** (Newton's third law).
3. Thrust must exceed **9.8 m/s²** of gravitational acceleration.
4. **Aerodynamic laws** govern the efficiency of flapping.
5. **Control logic** adjusts wing angle, flap frequency, and tail orientation to maintain balance and direction.

A useful workflow: **sense → analyze → decide → control → correct (continuously)**.

**Level 3 — Implementational (WHERE):**

The physical components required:

- **Wings** that can flap
- **Sensors** to detect height, orientation, and obstacles
- **Actuators** (motors) to drive the wings
- **A controller** (microprocessor) to execute the control logic
- **A power system** (battery)
- **A lightweight, aerodynamic body** to reduce friction

### 6.4 The Logical Chain: How the Levels Connect

The three levels are not independent. They form an explanatory chain:

> You have a **wing** (Level 3) → you **flap** it (Level 2) → *because of that* you generate **thrust** → *because of that* you achieve **flight** (Level 1).

The phrase **"because of that"** is the signature of explanation. It is how you know the levels are linked. Level 3 confirms Level 2, and Level 2 explains Level 1.

### 6.5 Top-Down Design, Bottom-Up Validation

A critical question: should you proceed from top to bottom, or bottom to top?

**The answer: both, in sequence.**

**Design: TOP → BOTTOM.**

1. Define the problem statement and goal (Level 1).
2. The goal suggests the algorithms required (Level 2).
3. The algorithms determine the physical components needed (Level 3).

**Validation: BOTTOM → TOP.**

1. Check that the hardware actually executes the rules.
2. Verify that sensors feed the controller correctly.
3. Confirm that the controller drives the actuators.
4. Ensure that the wings produce sufficient thrust.
5. Validate that the system achieves flight.

**Analogy:** Building a house — you design from the blueprint down (top-down), but you inspect by walking from the foundation up (bottom-up). You need both directions to trust the house.

### 6.6 State-Machine Perspective

A useful way to think about the robot's behavior is as a **state machine**:

| State | Transition | Mechanism |
|-------|-----------|-----------|
| Stationary on ground | → Ascend | Mechanics to stand, then flap wings to generate thrust |
| Ascending | → Flying | Continue flapping, adjust angle |
| Flying | → Descending | Reduce thrust, adjust wing angle |
| Descending | → Stationary on ground | Land, absorb impact |

At the computational level, you calculate the parameters: **weight, wing span, tail angle, head position**. At the algorithmic level, you define the control logic that transitions between states. At the implementational level, you build the sensors, motors, and wings that execute those transitions.

### 6.7 The Three Conclusions

This exercise teaches three things:

1. **Design information-processing systems from all three perspectives.** Do not stop at the algorithm.
2. **Distinguish the three levels clearly.** You can contribute to science or engineering at any single level — but you must know which level you are working at.
3. **Understand how the levels interrelate.** Top-down to develop, bottom-up to validate. This integration is what produces better AI.

---

## Part 7: What Is AI? Historical Definitions

The slide of historical definitions can be organized along two axes: **thinking** and **acting**.

| Axis | Definition Type | Example Definitions |
|------|----------------|---------------------|
| **Think** (like humans) | Mental faculties | "The exciting new effort to make computers think — machines with minds"; "The study of mental faculties through the use of computational models" |
| **Act** (like humans) | Behavior | "The study of how to make computers do things at which, at the moment, people are better" |

A third thread runs through some definitions: **rationality** — developing machines that are rational in the way humans are.

**Analogy for the two axes:** "Think" is caring *how* a chess engine reasons — does it reason like a human? "Act" is caring only that it *wins* — it behaves like (or better than) a human. The think-vs-act distinction will recur throughout the course.

---

## Part 8: The Turing Test and Its Limits

### 8.1 The Original Question and the Clever Mapping

Alan Turing's original question was: **"Can machines think?"**

But "think" is difficult to define. So Turing made a clever substitution: he mapped **"think" → "behave."** Since the test deals with communication, behavior is observable in a way that thinking is not.

### 8.2 The Thought Experiment

Imagine two rooms:

- In **Room A**, there is a human.
- In **Room B**, there is an undisclosed "someone" — either a human or a machine.
- A **human interrogator** communicates with both rooms via text.

If the interrogator **fails to distinguish** which room contains the machine, then the machine is deemed intelligent.

### 8.3 It Is an Operational Definition

The key insight: the Turing Test **does not define what thinking or intelligence IS**. It says: *if you achieve the same task and the interrogator cannot tell the difference, that counts as intelligence.*

This is an **operational definition** — a definition based on observable outcomes rather than internal states.

**Analogy:** A counterfeit bill is not defined by what money "is." It is defined operationally: *if an expert cannot tell it from real, it passes.* The Turing Test is the counterfeit test for minds.

The test is an **imitation game** — it measures copying of human behavior, not genuine understanding.

### 8.4 Limitations of the Classical Test

**Limitation 1: It is confined to two rooms.**

The real world is not two rooms. Intelligence in the real world involves navigating physical space, interacting with objects, and responding to a dynamic environment.

**Limitation 2: It is contingent on the interrogator.**

The test is person-specific. A different interrogator might reach a different conclusion. This introduces error and non-reproducibility.

**Limitation 3: It is text-only.**

The test involves only text conversation. But text has inherent limitations. How can text alone represent intelligence? Missing are:

- **Physical interaction** with the interrogator and environment
- **Handshaking, gestures**, and the physical behaviors humans use when communicating
- **Perception** of the world

### 8.5 The Total Turing Test

To truly judge intelligence, a system must pass the **Total Turing Test**. This requires:

- **Perception** — vision and other senses
- **Actuation** — limbs, hands, legs, motors
- **Interaction with the environment** — navigating the real world

**The shift:** From *"act like a human through conversation"* (two rooms) to *"humanly behave and interact with the world"* (the real world).

**Analogy:** The classical Turing Test is a **pen-pal exam** — letters only. The Total Turing Test is a **driving test** — you must see the road, move the wheel, and react to the world. A pen pal can fake being human; a driver cannot.

### 8.6 Implications for NLP

This is why **NLP is not restricted to text**. A natural language processing system that aims for true intelligence requires:

- **Vision** — to perceive the world
- **Robotics and actuators** — to act in the world
- **Audio processing** — to handle speech (which can be transcribed to text)

This course, Foundations of Vision & Language, aims toward the Total Turing Test: developing systems that are not contingent on or restricted to text, but that integrate vision (and eventually audio).

**What a computer needs to pass the Total Turing Test:**

- Natural language processing
- Knowledge retrieval
- Interaction with the world
- Vision techniques
- Actuators (as in the robot-flying exercise)

---

## Part 9: NLP Is Everywhere

### 9.1 Ubiquitous NLP Systems

NLP systems are **computational systems that process human language** for various applications. They surround us:

- Smart assistants (Alexa, Siri)
- Search engines (Google Search)
- Predictive analytics
- Machine translation (Google Translate)

These systems involve both **language** and **speech**, catering to different applications.

### 9.2 The Vending Machine as an NLP System

Consider a vending machine. How is it an NLP system?

- The machine has a **tab** displaying products. You **see** the product (vision), **match** it to what you want, select it, pay, and receive it.
- If equipped with a speech assistant, you **say** the product name, it **recognizes** the speech, and handles the transaction automatically.
- The pipeline is **speech-to-text → speech-to-behavior**.

**Analogy:** The vending machine is a **mini Total-Turing-Test arena** — vision (reading the tab) + language (product name or selection) + actuation (dispensing) all in one box.

A vision-enabled vending machine can even detect where you are looking or pointing. Smart AI technologies combine **all these human inputs** — vision, language, gesture — to interact with users.

---

## Part 10: Summary and Key Takeaways

### The Twelve Core Ideas

| # | Idea | Sticky Example |
|---|------|----------------|
| 1 | Language is tied to culture | Pirahã: grounded people, grounded language |
| 2 | Recursion = embedding clauses within clauses | Russian nesting dolls |
| 3 | Machines map words to numbers and search vocabularies | Librarian with index cards |
| 4 | Train computers like children | "Birth of a word" — language acquisition |
| 5 | Science and engineering perspectives together | Chef who can also explain the recipe |
| 6 | Chain: language → thought → thinking → information processing → cognitive mimicry → reverse engineering | The kitchen with five doors and a small counter |
| 7 | Marr's three levels: Computational (what/why), Algorithmic (how), Implementational (where) | Tea-making: goal, steps, kettle |
| 8 | Do not rush to method — understand the problem first | Robot-flying: define goal before motors |
| 9 | Top-down design, bottom-up validation | Blueprint down, foundation up |
| 10 | Turing Test = operational/imitation definition; limited by two rooms, interrogator, text-only | Counterfeit bill test vs. pen-pal exam |
| 11 | Total Turing Test needs perception + actuation + world interaction | Driving test, not pen-pal exam |
| 12 | NLP is everywhere; vending machine = vision + language + actuation | The mini Total-Turing arena |

### The One-Sentence Takeaway

> Language understanding is an information-processing problem that must be understood at three levels (what/why, how, where), validated in both directions (top-down design, bottom-up validation), and judged not by text tricks but by real-world perception and action — and the best way to build such systems is to think like a scientist and an engineer at once, treating the machine as a child that learns.

---

## Looking Ahead

In the next tutorial, we will:

- Debate the distinctions between **NLP** (Natural Language Processing), **NLU** (Natural Language Understanding), and **CL** (Computational Linguistics)
- Examine two historical NLU systems: **ELIZA** and **SHRDLU**
- Explore the **building blocks** of NLP and NLU
- Begin moving toward the engineering side of language systems

---

## Exercises

1. **Recursion practice:** Write a sentence with at least three levels of embedded clauses. Then rewrite it as a sequence of simple sentences without embedding. What is lost in translation?

2. **Marr's levels:** Choose a daily task (e.g., making coffee, driving to work, writing an email). Analyze it at all three of Marr's levels.

3. **Robot design:** Extend the robot-flying exercise. What additional sensors would you need for the robot to navigate around obstacles? At which of Marr's levels does this requirement appear?

4. **Turing Test critique:** List three questions you would ask in a Turing Test to distinguish a machine from a human. Then explain why a machine might fail or pass each one.

5. **NLP in the wild:** Find an NLP system you use daily. Identify its computational goal, its algorithmic approach (as best you can infer), and its implementational substrate.

---

*This tutorial is part of the Foundations of Vision & Language open-source course. Contributions and corrections are welcome via pull request.*