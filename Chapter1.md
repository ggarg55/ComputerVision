# Foundations of Vision and Language: A Complete Open-Source Tutorial on Computer Vision

Welcome! This tutorial is designed for anyone interested in computer vision. The goal is not memorization. The goal is understanding: how machines see, represent, recognize, reason, decide, and act.

Use this tutorial as a standalone guide, a revision tool, or a reference for building computer vision intuition.

---

## What Is Vision and Computer Vision?

**Vision** is how we understand the world from visual data: images, video, and 3D scenes.

**Computer vision** gives machines the ability to automatically understand images and video. It includes:

- Recognizing objects.
- Understanding relationships between objects.
- Recognizing people, scenes, and activities.
- Computing properties of the 3D world—actually 4D, because time is involved—from visual data.
- Using algorithms and representations so machines can perceive.

Computer vision comes under **perception**. Perception is different for different people because it is shaped by life experience.

**Simple example:**  
You see a crowded market. One person sees “people, mountains, buildings.” Another sees “a philosophical scene of human exploration.” Another sees “a noisy crowded place.” Same pixels, different perception.

Computer vision aims to give robots and autonomous agents the same kind of perception humans have.

---

## Recognition—Objects, Relationships, Scenes, and More

Recognition is not just “find the object.” It involves multiple levels:

- **Objects**: people, trees, mountains, buildings, street lamps, vendors.
- **Activities**: people roaming in the market.
- **Object detection and categorization**: multiple types of objects, not just one.
- **Instance recognition**: identifying that this is a particular location.
- **Scene and context categorization**: outdoor, within a city, crowded place.
- **Attribute recognition**: gray, made of fabric, crowded, flat.
- **Relationships between objects**: who is next to whom, what is behind what.
- **Function**: all chairs are chairs, but their designs differ.
- **Dynamics**: motion changes over time.

**Why recognition matters:**

- Robots and autonomous agents need it to see the world.
- It helps organize and give access to visual content.
- It connects visual data to information.
- It detects trends and themes.

**Simple example:**  
In a kitchen photo, you recognize not just “cup” and “table,” but “cup is on table,” “person is cooking,” “this is a kitchen,” and “the cup is red and ceramic.” That is recognition at multiple levels.

---

## History and Evolution of Computer Vision

Computer vision is not new. It has evolved over decades.

**Early examples:**

- Small cameras.
- Face detection when a person smiles.
- Identifying where a photo was taken.
- Visual queries.
- Community photo collections: people uploaded photos so AI models could have data.
- Autonomous agents detecting objects.

**Key milestone:**

- MIT, 1963: “Machine Perception of Three-Dimensional Solids.”
- Started with simple object boundaries.
- Now we have high-resolution dense images and millions of pixels.

**Data explosion:**

- Old data: 20 hours of video.
- Now: tens of thousands of hours of video added to platforms every minute.
- Thousands of new tagged photos added daily.
- Image size: first datasets were 32×32, processed at 16×16.
- Now images are 2048×2048 or larger.
- 30+ degrees of freedom in pose.
- 3,000 to 30,000 human-recognizable object categories.

**Simple example:**  
Early computer vision was like teaching a child to recognize a square. Today it is like teaching a child to recognize every object in a shopping mall, from every angle, in rain, night, crowd, and motion.

---

## Challenges in Computer Vision

Real-world computer vision is hard. Especially for autonomous cars.

**Key challenges:**

| Challenge | Simple Explanation |
|---|---|
| Real-time detection | Must detect a sudden animal instantly. |
| Decision latency | Must decide quickly and safely. |
| Trust | Humans must trust the system. |
| Heavy computing | On-board computers must be powerful. |
| Classification | Must know what the object is. |
| Domain shift | Trained on one dataset, tested on another. |
| Occlusion | One person visible, a child hidden behind. |
| Low light | Night vision, surveillance cameras. |
| Partial observability | Not everything is visible. |
| 2D field of view | Limited perspective. |
| Poor road infrastructure | Potholes, missing lane markings. |
| Policy threats | Regulations may restrict deployment. |
| Weather | Rain, fog, snow change appearance. |
| Imbalanced data | More summer data, less winter/rain data. |
| Generalization | Model becomes biased to one condition. |
| Illumination | Lighting changes appearance. |
| Object pose | Same object from different angles. |
| Clutter | Dense, messy scenes. |
| Intraclass appearance | Same object looks different. |
| Viewpoint | Where you see it from matters. |
| Context | Relationships and scene matter. |
| Function | What the object is used for. |
| Dynamics | Motion over time. |
| Scaling efficiency | Time, memory, and energy costs. |

**Simple example:**  
A self-driving car on an Indian road must detect a cow, a pothole, a child behind a cart, rain, low light, and a sudden lane change—all in real time, with limited battery and compute.

Robustness matters. If any one challenge is missing, the model may fail. For autonomous systems, one failure can be a blunder.

---

## Video, Time, and Scaling Efficiency

**Video** is a sequence of frames. Frames are images. Combining 2D images gives 3D—video.

Video adds **time scale**. Now:

- Objects move from one place to another.
- Consecutive frames must be synced.
- Algorithms must understand pixel-by-pixel information.
- High resolution + time = lots of computation.

**Efficiency has three costs:**

- Time cost.
- Memory cost.
- Energy cost.

**Striking comparison:**

- Human brain takes about **20 KB** for one image processing.
- AI model may take **20,000 K**.
- AI is still not efficient.

**Scaling** means: if the algorithm is fast and energy-efficient, it can process more frames within limited time.

**Simple example:**  
A flipbook. If you flip pages slowly, you miss the motion. If you flip fast, you see motion. Video AI must flip pages fast while using very little battery.

---

## Learning with Limited Data

Data is not always abundant.

**Spectrum:**

- Unlabeled data, clutter, some labeled classes.
- Crop to objects, parts, classes, labels.
- More data → better performance.
- Less data → performance drops.

**Where limited data is common:**

- Medical records.
- Autism.
- Geolocations.
- Volcano.
- Rare diseases.
- Many cancer types.

In medical cases, augmentation techniques may not work well.

**Simple example:**  
There are millions of cat photos, but only a few hundred cases of a rare disease. AI must learn from very few examples. That is hard.

---

## Computer Vision in the AI Landscape

Computer vision has been central to AI:

- CNNs were designed for computer vision.
- Transformers were designed for NLP.
- Then came Vision Transformer.
- Now text and vision combine: CLIP learns from both text and image.
- Multimodal learning learns from multiple modalities and aims for best alignment.

**The brain–AI cycle:**

- **Brain is for AI**: study how the brain works to design AI.
- **AI is for brain**: AI helps us understand the brain better.

**AI, ML, DL, and Generative AI:**

- AI contains machine learning, deep learning, and many techniques: CNN, RNN, Transformer.
- Generative AI is a broad paradigm. GANs are a learning paradigm with generative and adversarial components, but they use deep learning techniques like CNN/Transformer.
- GANs learn an **implicit** data distribution, not an explicit one.

**Computer vision tasks:**

- Object detection.
- Classification.
- Scene understanding.
- Semantic scene segmentation.
- 3D reconstruction.
- Object tracking.
- Human pose estimation.
- Activity recognition.
- Visual question answering.

Vision is the most important source of information for the human brain and the entrance hall of AI.

**Simple example:**  
A robot in a room can use graph neural networks to understand hierarchy: room → table → cup → handle. Then it can act.

---

## Human Vision and Marr’s Framework

Why does human vision need multiple stages instead of directly recognizing a 3D object from a 2D image?

Because a 2D image is ambiguous and incomplete:

- Same object appears different under lighting, distance, viewpoint, background, occlusion.
- Intermediate stages extract reliable information: boundaries, surfaces, shape, depth.
- Pixels tell us what is visible.
- Edges tell us where structure begins and ends.
- Depth tells us how structures are arranged.
- 3D model tells us what the object is.

**Marr’s computational vision framework, 1970s:**

1. **Input image**.
2. **Primal sketch**: intensities, boundaries, edge images.
3. **2.5D sketch**: depth, surface orientation.
4. **3D model**: surface and volumetric primitives.

**Why 2.5D?**

- It is XY plus depth information.
- Depth is relative to the viewer’s viewpoint.
- It is not a full volume model.
- We are not reaching full 3D from all viewpoints.
- So it is “2 and a half D.”

**Simple example:**  
A cardboard cutout of a chair has height and width and some depth cues, but you cannot see the back, the inside, or the full 3D structure. That is 2.5D.

---

## Features, SIFT, HOG, and Gradients

**Features** are informative visual properties that help distinguish objects.

Examples:

- Pixel intensities.
- Edges.
- Mesh.
- Shape.
- Corners.
- Distinct information.

**A good feature point is:**

- Distinctive.
- Stable under scale, rotation, lighting, viewpoint.
- Reliable across changes.

**SIFT** is a classical feature detection algorithm.

- Detects distinguished features.
- Can track key points even if image is rotated, scaled, modified.
- Can recover transformations.
- Invariant to scale and rotation.

**HOG** = Histogram of Oriented Gradients.

- Uses gradient information.
- Gradient = rate of change.
- Gradient has magnitude and direction.
- Magnitude: high = more important, low = less important.
- Direction: tells alignment or misalignment.
- Used for object detection, human detection, optical flow.

**Simple example:**  
To recognize a face, you don’t remember every pixel. You remember eyes, nose, mouth—distinct features. SIFT finds those landmarks. HOG describes the directions of edges—like drawing arrows along the outline of a person.

---

## Do We Still Need Classical Algorithms?

Yes. Classical algorithms like SIFT and HOG still matter.

**Reasons:**

- Annotation.
- Low compute environments.
- Small devices.
- Thermal cameras.
- Shape detection.
- Handmade filters when you know the object.
- Early layers of CNN.
- They can help CNN detect better.

Overfitting is not a reason to reject CNNs. Overfitting has solutions: regularization, data augmentation, better design.

**HOG fails when:**

- Contrast is low.
- Shapes are irregular.
- Shapes are highly curved.
- Patterns are random.
- Occlusion or dense environment.

**Simple example:**  
A master key can open many doors, but a custom key is cheaper and faster for one specific door. HOG/SIFT are custom keys for low-power, specific vision tasks.

---

## Prerequisite Quick Checks

**Cross entropy loss:**

- Measures how wrong your confidence is.
- Binary cross entropy: P and 1−P.
- Categorical cross entropy: softmax over many classes.

**Can softmax be used for binary classification?**

- Yes, with two classes.
- Sigmoid is the binary special case.

**GAN: implicit or explicit?**

- GAN learns an implicit data distribution.
- It does not explicitly model the probability density.
- It learns by generating samples.

**Simple example:**  
Cross entropy is a penalty for wrong confidence. If you say “99% cat” and it is a dog, you pay a huge penalty.

---

## Memory Cheat Sheet

| Concept | Simple Memory Hook |
|---|---|
| Hybrid mode | Dine-in + delivery setup |
| Computer vision | Giving eyes to machines |
| Perception | Same market, different interpretations |
| Recognition | Objects, activities, scene, instance, attributes |
| Autonomous car challenge | Cow + pothole + rain + night + occlusion in real time |
| Domain shift | Trained on summer, tested in winter |
| Imbalanced data | More summer photos, fewer rain photos |
| Occlusion | Hand behind hand |
| Video | Flipbook of frames + time |
| Scaling efficiency | Fast, low memory, low energy, more frames |
| Brain vs AI energy | Brain sips 20 KB; AI gulps 20,000 K |
| Limited data | Rare disease, autism, volcano |
| Marr’s 2.5D | Cardboard cutout with depth, not full 3D sculpture |
| SIFT | Face landmarks stable under rotation |
| HOG | Arrows of edge directions |
| Gradient | Rate of change with magnitude and direction |
| Softmax for binary | Yes, with two classes; sigmoid is special case |
| GAN | Implicit distribution learning |
| Classical features still useful | Custom key for low-power specific tasks |
| HOG failures | Low contrast, irregular, curved, random, occlusion |
| Course mantra | Understand, don’t mug; think logically |

---

## Final Project Ideas

1. Build a low-light object detector and compare HOG vs CNN.
2. Create a mini self-driving car simulation with real-time detection.
3. Design a medical image classifier with limited data.
4. Implement SIFT and HOG for image matching.
5. Build a multimodal model that aligns text and images.
6. Create a video object tracker and measure time, memory, and energy.
7. Build a visual question answering demo.
8. Design a 2.5D depth estimation pipeline.

---

## Final Thoughts

Computer vision is not just about recognizing images. It is about building a robust, efficient, real-time perception system that can see like a human, reason like an engineer, and act safely in an uncertain world.