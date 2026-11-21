# Building Question Answering and Sentiment Analysis Systems

## A Practical Tutorial from First Principles

> **What you'll build in this tutorial**
> - **Part 1 — LibraryBot:** a question-answering system that takes a natural-language question, finds evidence in a document collection, and extracts the answer.
> - **Part 2 — ReviewSense:** a sentiment-analysis system that identifies *what* people feel and *about which aspect*.
>
> Along the way you'll learn the classical IR-based QA pipeline, the evaluation metrics, and the deep-learning architectures (RNN, TD-LSTM, TC-LSTM, attention) that power modern systems. Everything is explained with concrete examples and diagrams you can reproduce.

---

# Part 1 — Question Answering (QA)

## 1.1 What QA really tests

Consider the question:

> **"What do worms eat?"**

Several candidate answers are possible:

| Candidate answer | Correct? | Why |
|---|---|---|
| Worms eat grass. | ✅ | Directly answers the question |
| Horses with worms eat grass. | ❌ | Changes the subject to horses |
| Birds eat worms. | ❌ | Reverses the relation |
| Grass is eaten by worms. | ✅ | Same meaning, passive voice |

Only the green answers are correct. Notice that *"Worms eat grass"* and *"Grass is eaten by worms"* express the **same proposition** in different surface forms.

**This is the central challenge of QA:** it is a test of whether a computer can *understand* natural language — not merely match keywords.

> **Memory hook:** QA = **A-F-A** — **A**sk/understand, **F**ind evidence, **A**nswer/extract.

---

## 1.2 A short history of QA systems

QA as a formal task grew out of the **Text REtrieval Conference (TREC)**. The TREC QA track established a structured evaluation framework, and many modern systems trace their lineage back to it.

Three landmark systems illustrate the evolution:

### IBM's early competition-winning system

IBM's system did not always answer direct Q&A. Instead it solved **cloze-style** problems where the answer was hidden inside a passage:

- *"Where Peele stores his ball"* → **soccer locker**
- *"Before and after goes to the movie … film of a typical day in the life of the Beatles …"* → **A Hard Day's Night of the Living Dead**

The system had to locate the answer *between* the words of a sentence.

### Apple's Siri

Siri introduced one of the earliest **consumer-facing spoken QA** systems:

- *"Do I need an umbrella tomorrow in San Francisco?"* → *"Yes, San Francisco should get rain tomorrow."*

This demonstrated real-world QA on personal devices.

### Google Now and Wolfram Alpha

These brought QA-style interfaces to web-scale audiences.

**Takeaway:** QA moved from lab competitions → personal devices → web-scale assistants.

---

## 1.3 Two types of questions

QA datasets are usually divided into two broad categories.

### Factoid questions

Questions whose answers are short factual entities:

- *"Who wrote the Universal Declaration of Human Rights?"*
- *"How many calories are in two slices of apple pie?"*

Answers are typically **one to three words**. Most commercial deep-learning QA systems were built around factoid questions.

### Complex / narrative questions

Questions that require **reasoning** and a **long-form answer**:

- *"In children with acute febrile illness, what is the role of acetaminophen in reducing fever?"*

Answering these requires synthesizing evidence across sentences.

### Why analyze answer type and question type?

Before answering, it helps to know:

- **Answer type** — what kind of entity is the answer? (person, city, date, quantity …)
- **Question type** — is this a definition question? a math question? a list question?

Knowing the answer type narrows the search space and reduces noise. Knowing the question type tells you *how* to answer.

> **Example:** *"Who founded Virgin Airlines?"* → answer type = **person**. *"What Canadian city has the largest population?"* → answer type = **city**.

---

## 1.4 Two families of QA methods

| Family | How it works | Example inspiration |
|---|---|---|
| **IR-based** | Retrieve relevant **text passages** from large collections, then extract answers | Google, IBM Watson |
| **Knowledge-based** | Query **structured sources** such as knowledge graphs | Knowledge-graph QA |
| **Hybrid** | Combine both IR and structured knowledge | Modern systems |

The key difference: **IR searches text; KB searches structured facts.**

**Limitation of web search:** Search engines already answer many everyday factoid questions ("Who wrote Harry Potter?"). But web search is **not sufficient for deeper or contextual reasoning** — that is where structured retrieval and QA pipelines earn their keep.

---

## 1.5 The IR-based QA pipeline: three core steps

This is the backbone of classical QA. Learn it as **Q-D-A**:

```
┌─────────────────────┐     ┌──────────────────────┐     ┌─────────────────────┐
│  QUESTION           │     │  DOCUMENT            │     │  ANSWER             │
│  PROCESSING         │ ──▶ │  RETRIEVAL           │ ──▶ │  EXTRACTION         │
│                     │     │                      │     │                     │
│ • Query formulation │     │ • Keyword selection  │     │ • NER tagging       │
│ • Answer type       │     │ • Passage retrieval  │     │ • Candidate ranking │
│ • Question type     │     │ • Passage ranking    │     │                     │
│ • Focus detection   │     │                      │     │                     │
│ • Relation extract. │     │                      │     │                     │
└─────────────────────┘     └──────────────────────┘     └─────────────────────┘
```

> **Memory hook:** **Q-D-A** — **Q**uestion processing → **D**ocument retrieval → **A**nswer extraction.

### Step 1 — Question processing

This step *understands* the question. It includes:

- **Query formulation** — choose the keywords that matter.
  - For *"Who is the Prime Minister?"*, the keywords are **"Prime Minister"** — not every word.
- **Answer type detection** — predict the named-entity type of the answer (person, place, city, date …).
- **Question type classification** — definition? math? list?
- **Focus detection** — what exactly is being asked.
- **Relation extraction** *(optional)* — structure relations between entities.

#### Worked example: the Florida border question

> *"There are two states you could be re-entering if you are crossing Florida's northern border."*

| Component | Value |
|---|---|
| Answer type | US states |
| Query terms | two states |
| Focus | two states |
| Relation | borders Florida to the north |

This is called **structural analysis** — mapping question semantics to an **answer constant**.

A naive QA system might just embed the query and match it against documents. A fine-grained system additionally performs query formulation, answer-type detection, question-type classification, focus detection, and relation extraction.

#### Answer-type taxonomy

Answer types can be organized hierarchically:

```
Location
├── Country
├── State
├── City
└── …

Person
├── Politician
├── Athlete
└── …
```

Other common categories include animal, body, color, creative works, and so on. A hierarchical mapping makes answer-type prediction easier — the model can iterate over the taxonomy.

#### How to build an answer-type detector

Answer-type detection is simply a **classification task**. You can implement it with:

- Handwritten rules
- Classical machine learning
- A mix of rules + ML
- Deep learning

You define the taxonomy, annotate training data, and train a model.

### Step 2 — Document retrieval / passage retrieval

Once you understand the question, you retrieve evidence.

1. **Keyword selection** — decide which query/document keywords to use.
2. **IR engine retrieval** — retrieve documents using the filtered query terms.
3. **Segmentation** — split long documents into shorter units (paragraphs/chunks), because searching a full long document is difficult.
4. **Passage ranking** — rank passages by relevance: passage 1 is most important, passage 2 next, and so on.

#### Features for passage ranking

- **Embedding similarity** between query and passage
- **Attention score** — which is essentially cosine similarity
- **Word overlap** — number of overlapping words
- **Number of named entities** in the passage
- **Number of query words** present in the passage
- **Number of question n-grams** matched
- Traditional IR features

### Step 3 — Answer extraction / processing

Given the top-ranked passages, extract the answer.

- Run an **answer-type named-entity tagger** on the passage.
  - If the answer type is **city**, the tagger must detect cities.
  - If the answer type is **person**, the tagger must detect persons.

#### Worked example: "Who is the Prime Minister of India?"

- Answer type: **person**
- Passage: *"Manmohan Singh, Prime Minister of India, had told leaders that the deal would not be renegotiated."*
- NER tags **Manmohan Singh** as **person** → easy extraction.

#### Worked example: "How tall is Mount Everest?"

- Answer type: **height / length**
- Passage gives the official height → extract the quantity.

#### Handling multiple candidate answers

When several candidates are retrieved, **rank them** by probability/relevance. You can use ML, DL, or search-based ranking methods. For example:

> *"Who was Queen Victoria's second son?"* — answer type = person. The passage contains many person names. Ranking picks the most appropriate.

> **Memory hook for the full QA pipeline:**
> **Q-D-A** = Question processing → Document retrieval → Answer extraction.
> Question processing extras: **Q-A-Q-F-R** — **Q**uery formulation, **A**nswer type, **Q**uestion type, **F**ocus, **R**elation extraction.

---

## 1.6 Building a QA dataset from scratch

Suppose you want to build a QA system for a subject (say, a course on FOBL). Here is the recipe:

1. **Curate documents** — textbooks, lecture notes, papers on the subject. These become your document collection.
2. **Create questions** — ask students and the teacher for likely doubts. Five questions per student quickly yields a useful set.
3. **Implement the pipeline** — document extraction first, then answer retrieval.

### Connection to modern RAG and agents

**Retrieval-Augmented Generation (RAG)** systems and **LLM agents** are advanced versions of the same traditional QA pipeline:

- They maintain a **document index**.
- They **retrieve** relevant documents.
- An **LLM** answers or performs a task (e.g., email writing).

The backbone is unchanged: **retrieve → answer**.

---

## 1.7 Evaluating QA systems

### Accuracy

The simplest metric: does the system's answer **match the gold answer**?

### Mean Reciprocal Rank (MRR)

For each query, the system returns a **ranked list of m candidate answers**. Compare the system's ranking with the gold ranking.

> **Example:** The system ranks A3 first, A2 second, A4 third, … The gold ranking says A3 should be second. MRR gives credit based on **where the first correct answer appears** in the ranked list.

MRR rewards systems that place the correct answer near the top — exactly what you want in production QA.

---

## 1.8 Advanced structuring: relations, time, and geography

These techniques make answer retrieval **better and faster** by structuring data beyond raw text.

### Relation extraction

Extract structured relations between entities:

- *"Emma Goldman was born in June 27, 1989"* → relation: **born_in(Emma Goldman, 1989-06-27)**
- *"Modi is Prime Minister of India"* → relation: **prime_minister(Modi, India)**

This is one more level of **structuring** — it makes retrieval faster and more precise.

> **Nuance — "current":** For *"Who is the current Prime Minister of India?"*, you must encode **current** information into the entity/data structure. It's a data-preparation detail, but it matters.

### Temporal reasoning

> *"In 1594 he took a job as a tax collector in Indonesia."*

If the person was born in 1870, then *"took"* is a bad answer — he wasn't alive in 1594. *"Servant"* is possible because he was alive then. This is reasoning around **time**.

**LLMs are very poor at temporal reasoning.** Why? Time is tokenized poorly. There is no tokenizer that captures every way time appears:

- `26th June`
- `2606`
- `95`

Because of this, **evidence-based answering** and temporal reasoning remain open problems.

### Geospatial knowledge

Geographical QA needs structured knowledge about:

- Containment
- Directionality
- Borders

The **GeoNames** database (geonames.org) has almost all geographical information and is widely used for geographical QA.

---

## 1.9 Dense retrievers and contrastive learning

A **dense retriever** is a deep-learning retriever:

1. **Encode** the query into an embedding.
2. **Encode** each document/passage into an embedding.
3. **Score** pairs — cosine similarity, Manhattan distance, etc.
4. **Rank** passages by score.

Dense retrievers are used for:
- Document ranking
- Finding relevant passages
- Filtering irrelevant passages

### Contrastive learning

Contrastive learning is the standard training objective:

- **Positive passages** — contain the answer or are very close to it. Pull their embeddings **together** with the query.
- **Negative passages** — irrelevant. Push their embeddings **away**.
- **Hard negatives** — completely different domain. For example, a query about prime ministers paired with a passage about fairy tales. These are the hardest to separate and the most valuable training signal.

This mirrors the **negative sampling** idea from word embeddings.

---

# Part 2 — Sentiment Analysis

## 2.1 Why sentiment analysis exists

Humans want to know **what others think** to make decisions:

- Which car should I buy?
- Which school should I apply to?
- Which professor should I work for?
- Whom should I vote for?

So we collect reviews and feedback. But *too much feedback* creates a new problem: **how do you make a decision from it?** And not all feedback is trustworthy.

### The fake-review problem

Consider reviews on Yellow Pages for a moving company. Warning signs:

- Reviews posted on the **same date**
- Each reviewer reviewed **only one merchant**
- All reviews are **five-star**
- Reviewers know **exact employee names**
- **Too many positive mentions**

These signals suggest **fake or biased reviews**. Sentiment analysis is therefore not just "positive/negative" — it is also about **trust, subjectivity, and decision support**.

---

## 2.2 The subjectivity-analysis umbrella

Sentiment analysis sits inside **subjectivity analysis**, which includes:

- **Sentiment analysis**
- **Review mining**
- **Opinion mining**
- **Appraisal extraction**

These terms are often used interchangeably. *Appraisal extraction* is a more linguistic task, focused on evaluative language.

### What is subjectivity?

Subjectivity is the **linguistic expression of someone's opinion, sentiment, or emotion**. These are **private states**:

- Not open to objective verification
- Internal feelings, opinions, beliefs, emotions, attitudes
- Cannot be directly observed or objectively verified

Subjectivity analysis involves understanding blogs, editorials, reviews, and newspaper articles.

### Business perspective

Consider iPhone reviews. A company wants to know:

- What do consumers think of iPhones?
- What do they like and dislike?
- What are the major complaints?
- What features should we add?

This is **business intelligence** — a core concern in marketing and strategy. Product reviews are abundant, so understanding customer opinion matters for **marketing and strategy-building**.

### Modern applications

- **eBay** uses LLMs for product listing and description writing — but hallucination is a real problem.
- **Twitter as a data source:** if people are **calm** (low fear, low stress), the stock market tends to show **predictable movement after about 3 days**. Why the delay? Human emotions influence decisions, but the effect is not immediate: *mood → decision → impact*.

---

## 2.3 The four elements of sentiment

Sentiment analysis is the **detection of attitudes**. An attitude has four elements:

1. **Holder / source** — who expresses the opinion?
   - *"I love this phone"* → holder = **I**
   - *"Customers hate the service"* → holder = **customers**
2. **Target / aspect** — what is the opinion about?
   - *"I love this phone"* → target = **phone**
   - *"Battery is bad"* → target = **battery**
3. **Type of attitude** — what feeling or opinion?
   - **Fine-grained:** love, hate, like, desire, value
   - **Broad:** positive, negative, neutral
4. **Text containing the attitude** — sentence, document, review, blog, tweet

> **Memory hook:** **H-T-T-T** — **H**older, **T**arget, **T**ype, **T**ext.

### Aspect-based example

> *"The food was good but the service was bad."*

- Two aspects: **food**, **service**
- Two sentiments: **positive** for food, **negative** for service

This is **aspect-based sentiment analysis (ABSA)** — and it's essential because a single sentence can carry multiple sentiments about different aspects.

---

## 2.4 Task difficulty levels

| Level | Task |
|---|---|
| Simplest | Is the attitude positive or negative? |
| Moderate | Rank the attitude on a scale of 1–5 |
| Advanced | Detect the target, source, or complex attitude type |

### Early dataset: Polarity Detection

The early **polarity detection dataset** (IMDb movie reviews) contained examples like:

- *Star Wars review*: *"When Star Wars came out some 20 years ago, the image of traveling throughout the stars has become a commonplace image …"* → **positive**
- *Snake Eyes review*: *"Snake Eyes is the most aggravating kind of movie. The kind that shows so much potential but eventually becomes disappointing."* → **negative**

---

## 2.5 Data issues and hard problems

### Preprocessing challenges

- HTML/XML tags
- Twitter names, hashtags
- Phone numbers
- Emoticons

### Negation handling

Negation is a **classic, still-unresolved** problem. It can reverse sentiment:

- *"I didn't like this movie"* vs *"I really like this movie"*

To handle it, people preprocess data into forms like *"not like"*, *"not movie like this"* — so the model captures the correct polarity.

> **Subtle case:** *"I barely like this movie."* Humans read this as negative/down-toned, but a model may not treat it as negation. Exceptional cases like this require **specialized datasets and architectures**.

### Sarcasm and subtlety

Consider this perfume review:

> *"Because it is your darling fragrance, please wear it at home exclusively and tape the windows shut."*

The sentiment is **negative** — but there is no negative word in the text. Sarcasm detection is **still challenging**.

### Other hard problems

- **Ambiguous words**
- **Alternating sentiments** — positive → negative → positive within one document
- **Subject centrality** — which subject/aspect is being discussed?
- **Thought rate, expectation, ordering effects**

---

## 2.6 Sentiment lexicons

An early approach: collect **sentiment lexicons** — lists of positive and negative words.

| Lexicon | Notes |
|---|---|
| **LIWC** | Linguistic Inquiry and Word Count |
| **Bing Liu Opinion Lexicon** | Widely used |
| **SentiWordNet** | WordNet-based |

You can build lexicons per **language** (Hindi, English) and per **domain** (movie reviews, product reviews).

### Pros and cons

| Advantages | Disadvantages |
|---|---|
| Fast | Multiple word senses |
| No training data needed | Multiple word phrases |
| Good initial accuracy | Domain-specific constructions |

---

## 2.7 Aspect-Based Sentiment Analysis (ABSA)

This is the most important concept in this part.

> **Example:** *"Its battery is awesome but camera is very poor."*
> - Aspects: **battery**, **camera**
> - Sentiments: **positive**, **negative**

### Task 1 — Aspect term extraction

Identify all aspects in a sentence:

- *"I like the service and the staff but not the food."* → **service, staff, food**
- *"Ambience and music funky which I enjoy."* → **ambience, music**
- *"Awesome form factor and great battery life."* → **form factor, battery life**

Then define a **sentiment label per aspect**.

### Implicit aspects

Some aspects are not explicitly mentioned:

- *"It was very expensive for what you get."* → the aspect is **implicit** (probably price/value)
- *"I enjoy having Apple products."* → **Apple products** can be treated as an aspect

Some data contains **no aspect at all** — your corpus should include such cases.

### Coarse-grained vs fine-grained

| Granularity | Scope |
|---|---|
| **Coarse-grained** | Sentence-level: "This sentence is positive" |
| **Fine-grained** | Aspect-level: "Battery positive, camera negative" |

---

## 2.8 RNN models for sentiment

### Sentence-level RNN

```
"The zoom and display are awesome."
        │
        ▼
   ┌─────────┐
   │   RNN   │  → hidden states h₁, h₂, …, hₙ
   └─────────┘
        │
        ▼ last hidden state hₙ
   ┌─────────┐
   │ Linear  │  → scores
   └─────────┘
        │
        ▼
   positive / negative
```

The **last hidden state** is passed through a linear layer; the output scores determine polarity.

### The rise of LSTM-based ABSA (2016–2018)

Around 2016–2018, LSTM-based models for ABSA became very popular. The key insight: the model should **focus on the target aspect**.

### TD-LSTM — Target-Dependent LSTM

A plain LSTM models the sentence *without considering the target word being evaluated* — it treats every word equally.

**TD-LSTM** fixes this by running **two LSTMs** around the target:

```
Sentence:  "Its battery is awesome but camera is poor."
Target 1:  battery
           ┌──── LSTM ← left context ────┐   ┌──── LSTM → right context ────┐
           "Its"                          battery  "is awesome but camera is poor"
```

For target **battery**:
- LSTM 1: *"Its"*
- LSTM 2: *"battery is awesome but camera is poor"*

For target **camera**:
- LSTM 1: *"Its battery is awesome but camera"*
- LSTM 2: *"camera is poor"*

This explicitly highlights that there are **two aspect words** in the sentence.

### TC-LSTM — Target-Connection LSTM

**TC-LSTM** goes further by incorporating the **relationship between every word and the target**:

- Pair each word with the target: *"its battery"*, *"battery is awesome"*, *"camera battery"*, etc.
- Feed these pairs into the LSTM.

### Accuracy progression

```
plain LSTM  <  TD-LSTM  <  TC-LSTM
```

### Attention for ABSA

Use **attention** to focus on the important words/aspects:

- Design the attention function so it puts more weight on aspect words.
- Incorporate attention scores into the final prediction.

You can also **visualize attention weights** to check whether the model is focusing on the right target words.

---

## 2.9 From ABSA to fine-grained extraction

The field has progressed toward ever-finer extraction:

1. **Aspect sentiment extraction** — extract aspect + sentiment.
2. **Sentiment triplet extraction** — extract **aspect, opinion, polarity**.
   - *"Pizza is delicious"* → aspect = **pizza**, opinion = **delicious**, polarity = **positive**
3. **Quadruple extraction** — extract **aspect term, opinion term, sentiment polarity, aspect category**.

Each step adds a new level of structure, enabling more precise downstream applications.

---

## 2.10 Multilinguality and code-mixed text

Sentiment analysis in Hindi, code-mixed text, and transliterated text has its own challenges:

- **Out-of-vocabulary (OOV) words** — tokenization is hard
- Early work was English-only; multilingual methods soon followed
- **Bilingual word embeddings** — if you don't know a Hindi word's embedding, translate it to English and use the English embedding as an approximation

---

# Summary and Exam-Ready Takeaways

## QA — the essential checklist

| Concept | What to remember |
|---|---|
| **QA pipeline** | **Q-D-A**: Question processing → Document retrieval → Answer extraction |
| **Question processing** | **Q-A-Q-F-R**: Query formulation, Answer type, Question type, Focus, Relation extraction |
| **Retrieval** | Keyword selection → passage retrieval → passage ranking |
| **Answer extraction** | NER tagger matching answer type → rank candidate answers |
| **Data creation** | Curate documents → create questions → implement extraction & retrieval |
| **Evaluation** | Accuracy + MRR (rank of first correct answer) |
| **Advanced structuring** | Relation extraction, temporal reasoning, geospatial knowledge |
| **Dense retrievers** | Encode query + docs, score with cosine/Manhattan, train with contrastive learning (hard negatives!) |

## Sentiment — the essential checklist

| Concept | What to remember |
|---|---|
| **Four elements** | **H-T-T-T**: Holder, Target, Type, Text |
| **Subjectivity** | Umbrella: sentiment analysis, review mining, opinion mining, appraisal extraction |
| **ABSA** | Aspect first → sentiment second. Handle implicit aspects. |
| **Coarse vs fine** | Sentence-level vs aspect-level |
| **RNN baseline** | Last hidden state → linear → polarity |
| **TD-LSTM** | Left/right LSTM around the target |
| **TC-LSTM** | Every-word → target connection |
| **Attention** | Focus on aspect words; visualize weights |
| **Extraction levels** | Aspect → triplet (aspect, opinion, polarity) → quadruple |
| **Hard problems** | Negation, sarcasm, alternating sentiments, subject centrality, multilinguality |

## Formulas in exams

Formulas for standard components (attention, softmax, loss functions) will typically be provided. Focus your study on **architecture intuition** and **pipeline design** — not on memorizing LSTM cell equations, which are not expected in detail. **RNN with attention** is the architecture you should be comfortable with.

---

## Final mental model

> **QA** is a **retrieval + extraction** problem: understand the question, find evidence, extract and rank answers.
>
> **Sentiment analysis** is an **attitude detection** problem: identify holder, target, type, and text — and for ABSA, do it **per aspect**.
>
> Modern systems (RAG, agents, LLM-based assistants) are **advanced versions of these same pipelines**. Master the classical building blocks and the modern systems become easy to reason about.