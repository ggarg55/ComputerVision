# 🧠 Computer Vision & Natural Language Processing: A Complete Guide

Welcome to the **Computer Vision & Natural Language Processing Tutorial Series**! This repository is an open-source, comprehensive guide covering foundational theories, mathematical intuitions, classical algorithmic techniques, and modern deep-learning architectures across both visual perception and natural language understanding.

---

## 🧭 Repository Structure

```text
.
├── README.md
├── Part1/   # Computer Vision Foundations & Deep Learning (Chapters 1–8)
├── Part2/   # Language, Information Theory & Classical NLP (Chapters 1–10)
└── Part3/   # Neural NLP, Word Embeddings & Sequence Models (Chapters 1–8)
```

---

## 🎯 Learning Paths

You can study this curriculum sequentially or focus on your specialization:

* **Computer Vision Track:** [Part 1: Chapters 1 to 8](Part1/)
* **NLP & Information Theory Track:** [Part 2: Chapters 1 to 10](Part2/)
* **Deep Learning & Sequence Models Track:** [Part 3: Chapters 1 to 8](Part3/)
* **Multimodal AI Track:** Study all parts in order to understand how machines perceive and reason across both pixels and tokens.

---

## 📚 Curriculum Overview

### Part 1: Computer Vision Foundations & Deep Learning

Focuses on how machines represent, process, and extract semantic meaning from visual signals — from pixel-level point operators and filtering to modern convolutional and transformer-based dense prediction architectures.

| # | Chapter | Key Topics |
| :-: | :--- | :--- |
| **01** | [Foundations of Vision and Language](Part1/01_Foundations_of_Vision_and_Language.md) | Visual perception, biological vs. machine vision, representation levels, and logical intuition. |
| **02** | [Image Processing Foundations](Part1/02_Image_Processing_Foundations.md) | Histograms, brightness, contrast, dynamic range, spatial filtering, convolution, and noise removal. |
| **03** | [Computer Vision Essentials](Part1/03_Computer_Vision_Essentials.md) | 2D to 3D geometry, camera models, video basics, compression, and image restoration strategies. |
| **04** | [From Motion to Meaning](Part1/04_From_Motion_to_Meaning.md) | Optical flow, motion boundary histograms (MBH), histograms of optical flow (HOF), and action recognition. |
| **05** | [Semantic Segmentation: U-Net to Transformers](Part1/05_Semantic_Segmentation_UNet_to_Transformers.md) | Pixel-level classification, encoder-decoder networks, skip connections, U-Net, and Vision Transformers. |
| **06** | [Classification to Semantic Segmentation](Part1/06_Classification_to_Semantic_Segmentation.md) | Transitioning from image-level classification to dense spatial predictions and receptive fields. |
| **07** | [Object Detection: Two-Stage Detectors & Metrics](Part1/07_Object_Detection_Two_Stage_Detectors.md) | R-CNN, Fast R-CNN, Faster R-CNN, Region Proposal Networks (RPN), IoU, precision-recall, and mAP. |
| **08** | [Object Detection: Complete Tutorial](Part1/08_Object_Detection_Complete_Tutorial.md) | Modern detector design, one-stage vs. two-stage detectors, anchor boxes, and loss functions. |

---

### Part 2: Language, Information Theory & Classical NLP

Examines the nature of human language, linguistic representations, statistical regularity, information theory, statistical language modeling, and grammatical structures.

| # | Chapter | Key Topics |
| :-: | :--- | :--- |
| **01** | [What Is Language? Symbols, Meaning & Mind](Part2/01_What_Is_Language_Symbols_and_Mind.md) | Semiotics, symbolic grounding, language as a reconstruction protocol, and ambiguity. |
| **02** | [Language, Thought & The Three Levels of Analysis](Part2/02_Language_Thought_and_Levels_of_Analysis.md) | Pirahã language case study, Universal Grammar, Marr's three levels, and Turing tests. |
| **03** | [NLP, NLU & Computational Linguistics](Part2/03_Understanding_NLP_NLU_and_Computational_Linguistics.md) | Differences between NLP, NLU, and computational linguistics; taxonomy of core language tasks. |
| **04** | [Understanding Zipf's Law](Part2/04_Understanding_Zipfs_Law.md) | Power-law distributions in natural language, rank-frequency relationships, and vocabulary scaling. |
| **05** | [Quantifying Regularity in Language](Part2/05_Quantifying_Regularity_in_Language.md) | Foundations of redundancy, statistical regularity, and historical milestones from Turing to Shannon. |
| **06** | [Information Theory & Language](Part2/06_Information_Theory_and_Language.md) | Bridges Zipf's law to Shannon entropy, bits of information, communication channels, and coding theory. |
| **07** | [Information Theory for Language Models](Part2/07_Information_Theory_for_Language_Models.md) | Modeling sequence uncertainty, conditional entropy, mutual information, and predictive context. |
| **08** | [Information-Theoretic Measures for Language](Part2/08_Information_Theoretic_Measures_Entropy_to_Perplexity.md) | Shannon entropy, cross-entropy, relative entropy (KL divergence), and calculating perplexity. |
| **09** | [Practical Guide to N-Gram Language Models](Part2/09_Practical_Guide_to_NGram_Language_Models.md) | Maximum likelihood estimation, smoothing techniques (Good-Turing, Kneser-Ney), and SRILM walkthrough. |
| **10** | [POS Tagging & Syntactic Analysis](Part2/10_POS_Tagging_and_Syntactic_Analysis.md) | Part-of-speech tagsets, Hidden Markov Models, Viterbi decoding, dependency trees, and syntax parsing. |

---

### Part 3: Neural NLP, Word Embeddings & Sequence Models

Covers the modern deep-learning revolution in NLP: from continuous vector representations (Word2Vec, GloVe) to recurrent networks, attention mechanisms, and complete NLP applications.

| # | Chapter | Key Topics |
| :-: | :--- | :--- |
| **01** | [Neural Networks for NLP: Linear Models to Vectors](Part3/01_Neural_Networks_for_NLP_Linear_Regression_to_Word_Vectors.md) | Transition from linear regression/logistic classification to continuous distributed representations. |
| **02** | [Word Embeddings, Negative Sampling & GloVe](Part3/02_Word_Embeddings_Negative_Sampling_and_GloVe.md) | Word2Vec scaling bottlenecks, negative sampling mathematics, co-occurrence statistics, and GloVe. |
| **03** | [Word Embeddings and Word2Vec](Part3/03_Word_Embeddings_and_Word2Vec.md) | Why one-hot encodings fail, the distributional hypothesis, Skip-Gram vs. CBOW, and toy implementation. |
| **04** | [Word Embeddings Evaluation & Language Models](Part3/04_Word_Embeddings_Evaluation_and_Language_Models.md) | Intrinsic vs. extrinsic evaluation, word analogies, cosine similarity, and building PyTorch language models. |
| **05** | [Neural Networks & Language Models: Gradients to RNNs](Part3/05_Neural_Networks_and_Language_Models_Gradients_to_RNNs.md) | Gradient descent dynamics, window-based neural language models, and introduction to recurrent memory. |
| **06** | [From RNNs to Attention: Conceptual Architecture](Part3/06_From_RNNs_to_Attention_Complete_Tutorial.md) | Bidirectional RNNs, vanishing/exploding gradients, LSTM & GRU gates, and encoder-decoder bottlenecks. |
| **07** | [From RNNs to Attention: Practical Implementation](Part3/07_From_RNNs_to_Attention_Practical_Tutorial.md) | Sequence-to-sequence translation pipelines, dot-product attention mechanics, and transition to Transformers. |
| **08** | [Building QA and Sentiment Analysis Systems](Part3/08_Building_Question_Answering_and_Sentiment_Analysis.md) | End-to-end applications: IR-based question answering (LibraryBot) and aspect-based sentiment analysis (ReviewSense). |

---

## 🛠️ Usage & Prerequisites

* **Markdown Reader:** Files can be read directly on GitHub, inside VS Code/Antigravity, or using any Markdown viewer.
* **Math Rendering:** Equations are formatted in standard LaTeX/KaTeX (`$...$` and `$$...$$`).
* **Hands-on Implementations:** Many chapters include self-contained Python code blocks, PyTorch snippets, and algorithmic implementations.
