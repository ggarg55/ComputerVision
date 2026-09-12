# Object Detection: From Two-Stage Detectors to Evaluation Metrics

## A Complete Conceptual Tutorial

Object detection is one of the most important tasks in computer vision. It asks a machine to answer two questions at once:

1. **What is in the image?**
2. **Where is it?**

That “where” is usually a bounding box. So object detection is really:

> **Classification + Localization**

This tutorial walks through the full conceptual landscape of object detection: single-stage and two-stage detectors, Faster R-CNN, region proposals, multi-stage learning, interpolation, datasets, evaluation metrics, precision, recall, AP, mAP, architecture tradeoffs, and practical tips.

---

## 1. Two Ways to Build an Object Detector

There are two broad families of object detection methods.

### Single-Stage Detection

A single-stage detector looks at the image once and directly predicts objects and boxes.

**Simple example:**  
You glance at a room and say, “Dog on the left, cat on the right.”  
One shot. Fast. Sometimes less precise.

### Two-Stage Detection

A two-stage detector first makes rough guesses, then refines them.

**Simple example:**  
Stage 1: “Something is here. Something is there.”  
Stage 2: “That something is a dog, and the box should be tighter.”

Faster R-CNN is a classic two-stage detector.

- Stage 1 uses a **Region Proposal Network (RPN)**.
- It produces **class-agnostic proposals**.
- Stage 2 refines those proposals in feature space.
- It converts proposals into **fixed-size features**, then classifies them and improves the boxes.

The full pipeline looks like this:

> Image → Backbone Network → Region Proposal Network → Rough Proposals → Stage 2 → RoI Features → Fixed Size → Class + Refined Boxes

Here, **RoI** means **Region of Interest**.

---

## 2. Class Agnostic vs Task Agnostic

A key idea in two-stage detection is that Stage 1 is **class agnostic**.

- **Class agnostic** means: “We are not identifying the class. We are only finding boxes.”
- **Task agnostic** means: “We are not using any task ID or task-specific information.”

**Simple example:**  
A security guard says, “There is a person-shaped object.”  
He does **not** say, “That is Dr. Smith.”  
That is class agnostic.

In Faster R-CNN Stage 1:
- It says: “There is an object here.”
- It does **not** say: “This is a dog or a cat.”

Stage 2 then says: “This object is a dog, and the box should move slightly left.”

---

## 3. Multi-Stage Learning: Stage-Wise, Joint, and Hybrid

When a model has multiple stages, you can train it in different ways.

### A. Stage-Wise Training

Train Stage 1, then train Stage 2 separately.

**Simple example:**  
First learn alphabets. Then learn words.  
You do not let word mistakes change how you learned alphabets.

The assumption is that the stages are **not dependent** or **disconnected**.

### B. Joint Training

The stages are connected. Backpropagation goes from the last stage to the first stage.

**Simple example:**  
Semester 1 marks are highly useful to predict Semester 2 performance.  
So instead of optimizing each semester separately, optimize the whole degree end-to-end.

That is **joint training**.

### C. Hybrid Training

Do stage-wise training, then joint training. Or freeze Stage 1 and train only Stage 2.

**Simple example:**  
Learn alphabets separately. Then fine-tune reading as a whole.

### Losses in Faster R-CNN

In Faster R-CNN:

- Stage 1 has two losses:
  1. RPN classification loss
  2. RPN localization loss
- Stage 2 has two losses:
  1. Ground-truth-based object classification loss
  2. Object localization loss

So there are four losses in total.  
Joint training means all of them are optimized together.

---

## 4. Fixed-Size Features and Interpolation

Stage 2 needs proposals of different sizes to become fixed size.

Suppose you have a 256×256 image and want a 32×32 version.  
A naive approach is:

```python
image.resize(32, 32)
```

But this loses precision, important information, and relationships between objects.

**Simple example:**  
Take a class photo and shrink it to a thumbnail.  
Faces become blobs.  
But the relative positions still remain: teacher left, students right.

So we should not just reduce size directly.  
We use **interpolation**.

### Bilinear Interpolation

Interpolation means creating new data points from neighboring points.

**Simple example:**  
You have marks of nearby students.  
You need to guess a missing student’s marks.  
You take a weighted average of neighbors.

Bilinear interpolation does this for pixels.

- It creates a new pixel using neighboring pixels.
- It is a statistical method.
- No CNN, no convolution.
- The result is lower resolution and blurrier.
- But relative relationships between objects survive.

So a 32×32 image is lower quality than 256×256, but it is still useful.

---

## 5. Datasets for Object Detection

Before evaluating a detector, you need data.

Popular datasets:

- **Pascal VOC**: 20 classes, 5,000 images.
- **MS COCO**: 80 classes, 120,000 images. Widely used. If a good paper exists, it likely uses COCO.
- **Open Images**: a large-scale dataset.

If you plot the number of boxes per class, sorted by class frequency, you see a pattern:

- A few classes have many boxes.
- Many classes have very few boxes.

This is **class imbalance**.

A more severe version is the **long-tailed dataset problem**.

**Simple example:**  
Music streaming.  
A few songs are played millions of times.  
Many songs are played once.  
That is a long tail.

This is imbalance **across semantic categories**, not just foreground vs background.  
You are not talking within one image. You are talking across classes and their bounding boxes.

You must know what kind of dataset you are using.

---

## 6. Why Object Detection Evaluation Is Hard

Image classification is easy:

> Give one image → cat or dog → 75% accuracy.

Object detection is harder because it is **classification + bounding boxes**.

Problems that make evaluation complex:

### 1. Location Errors

Ground truth dog is a green box.  
Prediction is a magenta box.  
Is it correct or wrong?  
We need a rule.

### 2. Multiple Predictions for One Object

One dog gets many boxes.  
This is over-prediction.

### 3. Missed Objects

Two dogs are there, only one detected.  
This is under-prediction.

### 4. Two Objects in the Same Bounding Box

Not acceptable. This is also a false positive.

So evaluation must handle location, over-prediction, under-prediction, and missed objects.

---

## 7. IoU: Intersection over Union

IoU is the standard way to measure how well a predicted box matches a ground-truth box.

\[
\text{IoU} = \frac{\text{Area of Intersection}}{\text{Area of Union}}
\]

- If boxes are disjoint: IoU = 0.
- If boxes are identical: IoU = 1.
- If IoU > threshold: consider it correct.
- If IoU < threshold: consider it wrong.

**Simple example:**  
Two sticky notes on a table.  
Intersection = overlapping part.  
Union = total area covered by both.  
If overlap is more than half, you accept it.

Just like classification uses a probability threshold, detection uses an IoU threshold.

---

## 8. True/False Positives and Negatives

Understand the idea, not the table layout.

- **True Positive (TP)**: predicted positive, and it is true.
- **False Positive (FP)**: predicted positive, but it is wrong.
- **False Negative (FN)**: predicted negative, but it is actually positive.
- **True Negative (TN)**: predicted negative, and it is true.

**Simple spam example:**

- Spam = positive.
- Ham = negative.
- TP: spam correctly marked spam.
- FP: normal email marked spam.
- FN: spam marked normal.
- TN: normal email marked normal.

Different books draw the confusion matrix differently.  
0 and 1 can be swapped.  
Do not mug up the table. Understand the fundamental idea.

---

## 9. Precision and Recall

Precision:

\[
\text{Precision} = \frac{TP}{TP + FP}
\]

> Of all detections the model produced, how many were true?

Recall:

\[
\text{Recall} = \frac{TP}{TP + FN}
\]

> Of all real positives in ground truth, how many did the model detect?

Recall is also called **completeness**.

**Simple examples:**

- High precision: medical cases. Do not tell a healthy person they are sick.
- High recall: fraud detection, shoplifting. Do not miss a fraud. Better to flag a normal transaction than miss fraud.

Precision and recall are **inversely proportional**.

- Want high precision → recall drops.
- Want high recall → precision drops.
- Need a tradeoff.
- Optimal point is in between.

We plot a **Precision-Recall curve**.  
The area under the curve helps choose the threshold.

---

## 10. AP, mAP, and Multiple IoU Thresholds

For object detection:

- **AP** = Average Precision for each class independently.
- **mAP** = Mean Average Precision across classes.
- Then repeat with **multiple IoU thresholds**.

**Simple example:**  
Grade each subject separately → AP.  
Average all subjects → mAP.  
Then change the passing mark for each subject and average again.

This tells you which IoU threshold and which precision/recall threshold to choose.

---

## 11. Architectures and Practical Tips

Common object detection architectures:

- SSD
- Faster R-CNN
- R-FCN
- Inception
- Inception-ResNet
- MobileNet

COCO dataset results usually report mAP.

### How to Select a Model?

Evaluate multiple meta-architectures.  
There is no universal “this model works for this domain.”

### Resolution

Lower resolution is much faster.  
But it tends to miss smaller objects.  
It is blurrier.  
Fine-grained information is lost.

### Number of Proposals

Lower number of proposals is much faster.  
100 boxes vs 1000 boxes makes a big difference.  
But performance can drop when you increase proposals.

MobileNet is efficient.  
Inception-ResNet is bigger and often has higher mAP.  
So choose based on your resources.

### GPU Time

Higher mAP often costs more GPU time.  
Efficient models run faster but may have lower accuracy.

---

## 12. Pareto Optimal Curve

A graph of overall mAP vs GPU time often shows a **Pareto optimal curve**.

**Simple example:**  
A sports car is fast but cannot carry much.  
A truck carries much but is slow.  
You cannot maximize both without tradeoff.

If you want more accuracy, you add more layers and modules.  
That consumes more resources.  
You need an optimal point where efficiency and effectiveness are both acceptable.

---

## 13. Transfer Learning and MNIST

If you have limited resources:

- Take a pre-trained model.
- Transfer weights.
- Initialize from a model trained on another dataset.

**Simple example:**  
A chef trained in Italian cooking can learn Mexican cooking faster than starting from zero.

MNIST is a classic dataset:

- Black and white digits 0–9.
- 28×28 resolution.
- Very low resolution.
- Modern images can be 2048×2048.
- Larger is better.

So for detection, use larger, pre-trained models when possible.

---

## 14. What Comes Next

The next major topic after object detection is **semantic segmentation**.

Object detection tells you:
> “There is a dog here.”

Semantic segmentation tells you:
> “This pixel belongs to a dog, that pixel belongs to the background.”

It is a natural next step in visual understanding.

---

## Final Memory Map

| Concept | Simple Memory Hook |
|---|---|
| Single-stage | One glance: “dog left, cat right” |
| Two-stage | Rough boxes first, then label and refine |
| Class agnostic | “Something is here,” not “dog” |
| Task agnostic | “I don’t use task ID” |
| Stage-wise | Learn alphabets, then words separately |
| Joint training | Sem 1 marks predict Sem 2, optimize whole degree |
| Hybrid | Learn separately, then fine-tune together |
| RPN | Rough proposal machine |
| RoI | Region of Interest features |
| Bilinear interpolation | Guess new pixel from neighbors |
| Pascal VOC | 20 classes, 5K images |
| COCO | 80 classes, 120K images |
| Long tail | Few songs played millions, many played once |
| IoU | Sticky-note overlap |
| TP/FP/FN/TN | Spam vs ham |
| Precision | When alarm rings, how often real fire? |
| Recall | Of all real fires, how many alarms rang? |
| AP/mAP | Grade each subject, then average |
| Pareto | Speed vs capacity tradeoff |
| Transfer learning | Italian chef learns Mexican faster |
| Next | Semantic segmentation |

---