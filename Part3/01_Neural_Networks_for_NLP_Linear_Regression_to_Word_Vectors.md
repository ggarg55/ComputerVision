# Neural Networks for NLP: From Linear Regression to Word Vectors

## A Hands-On Foundation for Understanding Modern Language Models

---

## Who This Is For

If you want to understand how large language models actually work—not just use them—you need to start at the beginning. This tutorial builds neural networks from scratch using simple examples, then shows exactly how the same ideas apply to text. By the end, you will have implemented gradient descent, built a tiny neural network, and converted sentences into vectors that a network can consume.

No prior deep learning experience is assumed. Basic Python and high-school calculus (derivatives) will help.

---

## What You Will Build

1. A linear regression model that predicts dog tail wags from pats
2. A logistic regression classifier that predicts flower color from length and width
3. A gradient descent optimizer from scratch
4. A sentence classifier that converts text to one-hot vectors and passes them through a weight matrix

Along the way, you will understand **predicted values vs. gold values**, **cost functions**, **learning rates**, **partial derivatives**, **nonlinearities**, and **word representations**.

---

## Part 1: The Humble Line — Linear Regression

### The Dog-Patting Dataset

Imagine you have a dog. You pat the dog some number of times, and you count how many times the dog wags its tail. Your data looks like this:

| Pats (\(x\)) | Wags (\(y\)) |
|---|---|
| 1 | 2 |
| 2 | 4 |
| 4 | ? |

You want to predict the number of wags for a new number of pats. The simplest model is a straight line:

\[
\hat{y} = wx + b
\]

- \(w\) is the **weight**
- \(b\) is the **bias**
- \(\hat{y}\) is the **predicted value**

The given data points are called **training data**. The actual values you are trying to match are called **gold values** (or targets). These two terms—prediction and gold—are the most important vocabulary in all of neural networks.

### What Do Weight and Bias Actually Do?

Change \(w\) and the line **tilts**. Change \(b\) and the line **shifts up or down**. Together, \(w\) and \(b\) are called the **parameters** of the model. Learning means adjusting these parameters so the line fits the data as closely as possible.

> **Memory hook:** Weight tilts, bias shifts, gold is truth, prediction is guess.

---

## Part 2: Two Inputs and the Birth of Classification

### The Flower Dataset

Now suppose you have flowers described by two measurements: length and width. Each flower is labeled red or blue.

| Length | Width | Color |
|---|---|---|
| 3 | 1.5 | Red |
| 2 | 1 | Blue |

You want to predict the color of an unknown flower from its length and width. The linear equation becomes:

\[
z = w_1 m_1 + w_2 m_2 + b
\]

Here \(m_1\) and \(m_2\) are the inputs (length and width), and \(w_1, w_2\) are two weights.

But now the output \(z\) can be any real number—positive, negative, huge, tiny. That is not a label. We need a bounded output.

### The Sigmoid Function

The **sigmoid function** maps any real number to the interval \((0, 1)\):

\[
\sigma(z) = \frac{1}{1 + e^{-z}}
\]

- If \(z\) is large positive, \(\sigma(z) \approx 1\)
- If \(z\) is large negative, \(\sigma(z) \approx 0\)
- If \(z = 0\), \(\sigma(z) = 0.5\)

Why does this matter? Because values between 0 and 1 can be interpreted as **probabilities**. A prediction of \(0.92\) means "very likely red." A prediction of \(0.08\) means "very likely blue." You can apply a threshold at \(0.5\) to get a hard label.

Without sigmoid, a value like \(12\) or \(-7\) has no probabilistic meaning and no natural cutoff. Sigmoid gives you that cutoff.

This model—linear equation followed by sigmoid—is called **logistic regression**. It was the first approach to classification and remains a building block inside larger networks.

> **Memory hook:** Sigmoid turns scores into probabilities.

---

## Part 3: Cost Functions — Quantifying "How Wrong Am I?"

### The Error

Your model makes predictions. Your data has gold values. The difference between them is the **error**. But a single error number is not enough—you need a function that summarizes how wrong the model is across all examples. That function is the **cost function**.

The simplest workable cost function for regression is:

\[
C = (\hat{y} - y)^2
\]

where \(\hat{y}\) is the prediction and \(y\) is the target.

### Why Square?

Without squaring, the difference \(\hat{y} - y\) can range from \(-\infty\) to \(+\infty\). It can be large positive or large negative, and minimizing it is awkward. Squaring makes the function bowl-shaped: always non-negative, with a clear minimum.

### A Minimal Example

Strip the network down to a single bias parameter: \(n = b\). Suppose the target is \(4\). Then:

\[
C = (b - 4)^2
\]

From calculus, you can see the minimum is at \(b = 4\). But for real networks with millions of parameters and millions of data points, you cannot just "see" the answer. You need a method that finds the minimum without knowing it in advance. That method is **gradient descent**.

---

## Part 4: Gradient Descent — Walking Downhill in the Fog

### The Intuition

Imagine you are standing on a hilly surface in thick fog. You cannot see the bottom of the valley, but you can feel the slope beneath your feet. To reach the minimum, you step in the direction that goes downhill.

- If the slope is **positive**, you are on the right side of the valley—decrease the parameter.
- If the slope is **negative**, you are on the left side—increase the parameter.
- If the slope is **zero**, you are at the minimum (for this minimization problem).

The slope is the **derivative** of the cost function with respect to the parameter.

### The Update Rule

\[
b \leftarrow b - \alpha \frac{dC}{db}
\]

- \(\frac{dC}{db}\) is the gradient (slope)
- \(\alpha\) is the **learning rate**—how big a step you take

### Learning Rate: Too Big, Too Small

- **Too large:** You overshoot the minimum. The parameter bounces around or diverges.
- **Too small:** You converge very slowly. Training takes forever.
- **Just right:** You move smoothly toward the minimum.

> **Memory hook:** Gradient gives direction, learning rate gives courage.

### From One Parameter to Many: Partial Derivatives

When the cost depends on both \(w\) and \(b\), the cost surface is a 3D bowl. To find the gradient with respect to \(w\), treat \(b\) as a constant and differentiate. To find the gradient with respect to \(b\), treat \(w\) as a constant. These are **partial derivatives**.

\[
w \leftarrow w - \alpha \frac{\partial C}{\partial w}
\]
\[
b \leftarrow b - \alpha \frac{\partial C}{\partial b}
\]

The gradient points in the direction of steepest ascent. Subtracting it moves you downhill.

### Local Minima

Real cost surfaces can have many valleys—one global minimum and several local minima. Gradient descent can get stuck in a local minimum. Later optimization methods (momentum, Adam, etc.) address this, but the core idea remains: use the slope to decide direction.

---

## Part 5: Nonlinearities — Why Neural Networks Can Bend

### The Problem with Pure Linearity

If every layer is just \(y = wx + b\), stacking layers is pointless. A composition of linear functions is still a linear function. You get a straight line (or a flat plane) no matter how many layers you add.

**Nonlinearities** break this. They let the network bend, curve, and fit complex data.

### Common Nonlinearities

| Function | Range | Typical Use |
|---|---|---|
| Sigmoid | \((0, 1)\) | Binary classification |
| Tanh | \((-1, 1)\) | Hidden layers, training stability |
| Softmax | Simplex (sums to 1) | Multi-class classification |
| ReLU / Hard threshold | \([0, \infty)\) | Hidden layers, deep networks |

### Underfitting, Good Fit, Overfitting

- **Underfitting:** The model is too simple. It cannot capture the pattern.
- **Good fit:** The model captures the pattern without memorizing noise.
- **Overfitting:** The model memorizes the training data and fails on new data.

Nonlinearities give the network the capacity to fit curves. With great capacity comes the risk of overfitting—which is why regularization, validation, and early stopping exist.

> **Memory hook:** Linear gives a ruler; nonlinear gives a rubber band.

---

## Part 6: Hands-On Lab — Build and Break a Tiny Network

### The Setup

You have a dataset of flower measurements: length, width, and a binary label. You have a neural network with two weights and one bias. Gradient descent is implemented. The cost values are plotted at every epoch.

### Exercises

1. **Change the mystery flower dimensions.** Start with \((4.5, 1.0)\). Change the values and observe whether the predicted label changes.

2. **Reduce iterations from 10,000 to 100.** Observe the cost curve. Does it reach the minimum? Why or why not?

3. **Change the learning rate from 0.1 to 0.001.** Observe the cost curve. It becomes much smoother—almost linear—because the steps are tiny. With too few iterations, you never reach the minimum.

4. **Remove the square from the cost function.** Replace \((\text{prediction} - \text{target})^2\) with \((\text{prediction} - \text{target})\). Observe how the gradient changes. The derivative of the squared cost is \(2 \times (\text{prediction} - \text{target})\); removing the square changes the update dynamics.

5. **Inspect the gradient line.** Find the expression \(2 \times (\text{prediction} - \text{target})\). That is the derivative of the squared cost with respect to the prediction. This is the heart of backpropagation in miniature.

These exercises build intuition for how learning rate, iteration count, and cost function shape affect training.

---

## Part 7: From Flowers to Words — The NLP Pipeline

### The Task: Sentence Labeling

Replace the flower data with six short sentences, each labeled 0 or 1. For example:

```python
sentences = [
    "I like computer science",
    "I love NLP",
    "The dog wags its tail",
    "Neural networks are fun",
    "Gradient descent finds minima",
    "Words become vectors"
]
labels = [0, 1, 0, 1, 0, 1]
```

### Step 1: Tokenization

Split each sentence into tokens. The simplest method is whitespace splitting:

```python
tokens = [sentence.split() for sentence in sentences]
```

### Step 2: Build the Vocabulary

Collect all unique words across all sentences:

```python
vocab = sorted(set(word for sentence in tokens for word in sentence))
```

If "I" appears in every sentence, keep only one instance of "I." The vocabulary is the set of unique words.

### Step 3: Why Vectors?

A neural network is a group of mathematical equations. Equations take numbers as input. Words are not numbers. Therefore, every word must be converted into a numeric vector. This vector is called a **word representation** or **word embedding**.

### Step 4: One-Hot Encoding

The simplest word representation is **one-hot encoding**. Given a vocabulary of size \(V\), each word is represented by a vector of length \(V\) with a 1 in the position corresponding to that word and 0 everywhere else.

**Example:**

Vocabulary: `["New", "York", "Times", "Los", "Angeles"]` (size 5)

| Word | Vector |
|---|---|
| New | `[1, 0, 0, 0, 0]` |
| York | `[0, 1, 0, 0, 0]` |
| Times | `[0, 0, 1, 0, 0]` |
| Los | `[0, 0, 0, 1, 0]` |
| Angeles | `[0, 0, 0, 0, 1]` |

For a **sentence**, the vector has length \(V\) and contains a 1 for every word present in the sentence:

| Sentence | Vector |
|---|---|
| "New York Times" | `[1, 1, 1, 0, 0]` |
| "Los Angeles Times" | `[0, 0, 1, 1, 1]` |

> **Memory hook:** One-hot = tick the words present; vocabulary size = vector length.

### Step 5: Word Vector vs. Sentence Vector

- A **word vector** represents a single word. Its dimension is \(V\).
- A **sentence vector** represents a whole sentence. Its dimension is also \(V\), with 1s marking which words appear.

Both are one-hot representations. Neither captures word order or semantic similarity—but they are the simplest numeric form a network can consume.

---

## Part 8: The Matrix Neural Network

### From Vectors to Matrices

With multiple sentences, stack their vectors into a matrix \(H\):

- If you have \(N\) sentences and vocabulary size \(V\), then \(H\) is an \(N \times V\) matrix.
- Each row is one sentence's one-hot vector.

### The Weight Matrix

A neural network layer is defined by a weight matrix \(W\). If you want the layer to produce \(K\) outputs per sentence, then:

\[
W \text{ is } V \times K
\]

The output matrix is:

\[
S = H \times W
\]

The output \(S\) has shape \(N \times K\): one row per sentence, one column per output neuron.

### Example Dimensions

- 3 sentences, vocabulary size 5 → \(H\) is \(3 \times 5\)
- Want 2 output neurons → \(W\) is \(5 \times 2\)
- Output \(S\) is \(3 \times 2\)

### What Does \(H\) Do?

\(H\) is a **lookup matrix**. Given a sentence, it retrieves the corresponding one-hot vector. In code, this is often implemented as an embedding lookup: pick row \(i\) of \(H\) for sentence \(i\).

### What Does \(W\) Do?

\(W\) contains the learnable parameters. It is initialized randomly. During training, gradient descent updates \(W\) to minimize the cost function.

### The Fully Connected View

Picture a network with \(V\) input neurons (one per vocabulary word), a hidden layer of size \(K\), and an output layer. Every input neuron connects to every hidden neuron. The connection from input \(i\) to hidden \(j\) has weight \(W_{ij}\). That is exactly the matrix \(W\).

The output of hidden neuron \(j\) for sentence \(s\) is:

\[
S_{sj} = \sum_{i=1}^{V} H_{si} W_{ij}
\]

This is matrix multiplication. A neural network, at its core, is a sequence of matrix multiplications interspersed with nonlinearities.

> **Memory hook:** H looks up; W learns; S = H × W; loss corrects.

---

## Part 9: What Comes Next

### Cross-Entropy Loss

The squared-error cost used for regression is not appropriate for classification. For classification, the standard loss is **cross-entropy**:

\[
C = -\sum_{i} y_i \log(\hat{y}_i)
\]

where \(y_i\) is the true label (one-hot) and \(\hat{y}_i\) is the predicted probability. Cross-entropy penalizes confident wrong predictions heavily and is used almost everywhere in modern NLP.

### Word Embeddings: Skip-Gram

One-hot vectors are sparse and carry no semantic meaning. **Word embeddings** learn dense vectors where similar words have similar representations. The **skip-gram model** learns embeddings by predicting context words from a target word.

Given a sentence like "the cat sat on the mat," skip-gram uses "cat" to predict "the," "sat," etc. After training, words that appear in similar contexts end up with similar vectors.

Because you now understand matrices and gradient descent, the skip-gram model will feel like a natural extension: it is still a neural network, still trained with gradient descent, still using a cost function—just with a different architecture and objective.

---

## Summary: The One-Paragraph Version

A neural network is a chain of equations: turn data into numbers, multiply by weights, add bias, apply a nonlinearity, compare prediction with gold using a cost function, find the slope with gradients (partial derivatives for multiple parameters), and nudge parameters with a learning rate. In NLP, the first equation is special: words must become vectors, often one-hot vectors built from a vocabulary. Then sentences become matrices, weights are random matrices, and \(S = H \times W\) is the first neural layer. Modern LLMs are this same story at enormous scale.

---

## Key Terms

| Term | Meaning |
|---|---|
| Weight (\(w\)) | Parameter that tilts the line |
| Bias (\(b\)) | Parameter that shifts the line |
| Predicted value | Model's output |
| Gold value | True target |
| Cost function | Measures how wrong the model is |
| Gradient descent | Walks downhill using the slope |
| Learning rate (\(\alpha\)) | Step size |
| Partial derivative | Derivative with respect to one variable, others constant |
| Nonlinearity | Function that lets the network bend |
| Sigmoid | Maps reals to \((0,1)\) |
| One-hot vector | Vector with a single 1 and rest 0s |
| Vocabulary | Set of unique words |
| Tokenization | Splitting text into tokens |
| Word representation | Numeric vector for a word |
| \(H\) matrix | Sentence lookup / representation matrix |
| \(W\) matrix | Learnable weight matrix |
| \(S = H \times W\) | Output of a neural layer |

---

## Exercises

1. **Dog-patting regression:** Implement linear regression for the pats/wags data. Plot the line for \(w=0.25, b=0.68\). Compute predictions for \(x=1,2,4\). Compare with gold values.

2. **Flower classification:** Implement logistic regression for the length/width/color data. Use sigmoid. Try thresholds other than 0.5 and observe the effect.

3. **Gradient descent from scratch:** For \(C = (b-4)^2\), implement the update \(b \leftarrow b - \alpha \frac{dC}{db}\). Try \(\alpha = 0.1, 0.01, 1.0\). Plot the cost curve for each.

4. **Remove the square:** Change the cost to \(|\hat{y} - y|\) and observe how the gradient changes. Why is the squared cost easier to optimize?

5. **One-hot encoding:** For the vocabulary `["New", "York", "Times", "Los", "Angeles"]`, write the one-hot vectors for "New York Times" and "Los Angeles Times."

6. **Matrix dimensions:** For 6 sentences and a vocabulary of 26 words, what is the shape of \(H\)? If you want 3 output neurons, what is the shape of \(W\)? What is the shape of \(S\)?

7. **Build the network:** Implement \(S = H \times W\) in NumPy. Initialize \(W\) randomly. Print the output matrix. Verify that the dimensions match your answer to Exercise 6.

---

## Further Reading

- **Cross-entropy loss:** Understand why it is preferred over squared error for classification.
- **Skip-gram and word2vec:** Learn how dense word embeddings are trained.
- **Backpropagation:** The algorithm that computes gradients through multiple layers.
- **Optimizers:** Momentum, RMSProp, Adam—improvements over vanilla gradient descent.

---

*This tutorial is self-contained. All code can be run in a Jupyter notebook or Google Colab. Maintain your work in a public GitHub repository so others can learn from your implementations.*