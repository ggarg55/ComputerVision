# Computer Vision Essentials: From 2D Images to 3D, Video, and Restoration
---

## Learning Objectives

By the end of this tutorial, you will be able to:

- Explain how multiple 2D images become a 3D reconstruction.
- Understand the data and architecture revolutions behind image classification.
- Differentiate semantic segmentation from instance segmentation.
- Describe how CLIP connects images and text using contrastive learning.
- Understand text-to-image generation with DALL-E.
- Represent images as pixels, dimensions, and color models.
- Handle occlusion using finer resolution or multiple cameras.
- Apply image and video compression concepts.
- Understand I, P, and B frames in video compression.
- Use AI for image restoration in space, medical, and real-world domains.
- Build a learning strategy for computer vision that prioritizes thinking over syntax.

---

## 1. 3D Reconstruction: Many 2D Photos → One 3D World

### The Problem

You have many 2D photos of the same object or place. They come from:

- Different viewpoints
- Different people
- Different lighting and brightness
- Different hardware

Yet the goal is to reconstruct a single **coherent, seamless 3D image**.

### The Pipeline

1. Find similar or identical points/features across images.
2. Correlate those features.
3. Estimate camera position and viewing angle.
4. Merge all information into a 3D model.

A landmark 2009 ICCV paper by Agarwal et al. was a paradigm shift in this area. It showed how to incorporate **camera positions** and **viewing angles** to make 3D reconstruction more seamless.

### Simple Example

Imagine 20 tourists photographing the Taj Mahal from different angles. If you can match the same dome corners in all photos and know where each camera stood, you can rebuild a 3D model of the Taj.

### Key Takeaway

> 3D reconstruction = match features + know camera pose + merge.

---

## 2. Image Classification and the Data Revolution

### ImageNet and Fei-Fei Li

Fei-Fei Li is often called the **godmother of AI** for her role in creating ImageNet. The dataset started in 2004. Images were collected from the web, categorized, and used for object classification.

Today, ImageNet has:

- ~22,000 categories
- More than a million images

The **ILSVRC** challenge used 1,000 object classes and a massive number of images. It was challenging because of:

- **Interclass variation:** different classes look similar. Example: dog vs. wolf.
- **Intraclass variation:** the same class looks different. Example: the same dog sitting, running, wet, dark, or upside down.

### Data → Architecture Revolution

When data became huge, small models failed. This is **underfitting** — the model does not have enough capability.

So architectures improved:

- AlexNet
- GoogleNet / Inception
- VGG
- ResNet
- Vision Transformer (ViT)

**Turning point:** In 2017, model error became less than human error.

### Simple Example

If you give a child 10 flashcards, they may memorize. If you give them 1 million flashcards, they need a better brain. That “better brain” is the architecture.

### Key Takeaway

> AlexNet, ResNet, Inception, and ViT were turning points in computer vision.

---

## 3. Object Detection and Segmentation

### Object Detection

Object detection finds and localizes objects in an image.

### Segmentation

Segmentation assigns a label to every pixel. There are two main types:

- **Semantic segmentation:** segment by category.
- **Instance segmentation:** segment each individual object, even within the same category.

### Simple Example — Chairs

- **Semantic:** all chairs are colored blue. The model says “these pixels are chair.”
- **Instance:** chair 1 is red, chair 2 is blue, chair 3 is green. The model says “this is chair 1, this is chair 2.”

### Real-World Use Cases

- Manuscript reading: understand layout.
- Color segmentation for recognizers and detectors.
- Medical imaging.
- Legal documents: italic, bold, and highlighted text have different meanings. Segmentation can preserve hierarchy and importance.
- Memory-based documents: important parts are bold or highlighted, less important parts use smaller fonts, stored as images, then rendering extracts information.
- Autonomous vehicles: segment at coarse and fine levels.

### Key Takeaway

> Semantic = same class, same color.  
> Instance = same class, different objects, different colors.

---

## 4. CLIP: Connecting Images and Text

### The Question

Can we use image and text modalities together for better image classification?

### The Answer: CLIP

CLIP stands for **Contrastive Language-Image Pretraining**. It uses:

- A text encoder
- An image encoder
- Comparison of text embeddings and image embeddings

### Contrastive Learning

- Positive pairs: similar things together.
- Negative pairs: dissimilar things apart.
- Use contrastive loss.
- Maximize positive pair values.
- Minimize negative pair values.

### Training Example

Image I1 of a dog. Text T1 = “a dog.” This pair should have the highest value. Other texts like “a car” should be far away.

### Inference Example

Give an image. Compare it with class texts: “dog,” “cat,” “car.” The highest similarity wins. If “dog” has the highest value, classify as dog.

### Zero-Shot Classification

You do not give examples of every class. The model generalizes to anything. Related terms: zero-shot, one-shot, two-shot.

### SAM vs. CLIP

- **SAM (Segment Anything Model):** segmentation model, high-end fine-grain instance segmentation, trained differently, uses image + prompt/text.
- **CLIP:** classification. Different task.

### Key Takeaway

> CLIP = match image to text like a dating app for embeddings. Best match wins.

---

## 5. DALL-E: Text → Image

DALL-E creates images from text. ChatGPT uses DALL-E. Performance is tremendous now, but early versions had many issues. A good research direction is to study whether those problems still exist.

### Simple Example

Type “a cat riding a bicycle in space” → DALL-E generates the image.

### Key Takeaway

> CLIP goes image ↔ text for classification.  
> DALL-E goes text → image generation.

---

## 6. Image Representation and Color Models

### How Computers See Images

- Every pixel ranges from 0 to 255.
- Image size like 8×8, 16×16, 32×32 = height × width.
- A 2D image has height and width.
- Add color: R, G, B. So one image is 3D: 16×16×3.
- Color codes: RGB and BGR. This matters when programming. If you swap them, a red face becomes blue.
- Medical and food imaging may have high dimensions like 32 or 100.

### Color Models

- **RGB:** used for monitors, TVs, digital displays. Adds light to make white. Starts black, adds light → white.
- **CMYK:** used for printing. Opposite of RGB. Starts white, adds darkness → black.
- **HSV/HSL:** hue, saturation, lightness/value. Alternative color representation.

### Simple Example

- Phone screen = RGB. Turn off lights → black. Add red, green, blue light → white.
- Printer = CMYK. White paper + cyan, magenta, yellow, black ink → dark.

### Key Takeaway

> RGB = light, additive.  
> CMYK = ink, subtractive.

---

## 7. Occlusion

Occlusion happens when one object blocks another. Pixels overlap, and the 0–255 gradation differs.

### What Can Be Done?

- Instead of coarse 16×16 division, use finer division like 512×512. A traffic prediction paper used this to break occlusion.
- If an object is completely hidden behind another, a human cannot find it. It is as good as not existing.
- Single camera, single viewpoint: very difficult.
- **Collaborative edge computing:** multiple cameras with computing distribute information and process together. This handles occlusion.

### Simple Example

In a crowd, if one person is fully behind a wall, no single camera can see them. But multiple cameras from different angles can reconstruct them.

### Key Takeaway

> Occlusion = hidden pixels.  
> Fix with finer resolution or multiple views.

---

## 8. Image Compression

### Why Compress?

- Transfer data.
- Reduce size.
- Easier processing.
- Reduce processing power and footprint.
- Lower storage, faster transmission.
- Models need same size images.
- Limited hardware capability.
- Phone asks HD or not.

### Real-World Example: Surveillance

Police in Hyderabad spend lots of money sending surveillance video to Amazon servers outside India. Problems:

- Fast motion causes missing instances.
- Every frame goes to server → high cost.
- No local processing at camera side.
- Want inbuilt knowledge: if nothing is happening, don’t send the frame or send little info.

### How Would a Human Compress?

- Convert to grayscale.
- Histogram equalization.
- Reduce vector/pixel size but keep relevant details.
- Compress unwanted information.
- PCA can find dimensions with maximum information. But principal components are inferred, not actual data.
- Object detection: transfer only frames with objects of interest.

### Frequency-Space Method

Convert image to frequency space. Separate high and low frequency. Keep low frequency, significantly reduce high frequency to minimize information loss.

### PNG vs. JPG

- **PNG:** lossless. Better quality.
- **JPG:** lossy. Smoother because intensity is removed.
- Byte size depends on settings; the core distinction is lossless vs. lossy.

### Simple Example

Packing a suitcase. You keep essentials, fold clothes, remove extras. Compression = smaller suitcase, but you may lose some unimportant details.

### Key Takeaway

> Compression is not normalization.  
> Keep important information, reduce the rest.

---

## 9. Video Compression: I, P, and B Frames

### Video = Sequences of Images

Video adds time to images. Instead of storing every frame, store some full frames and predict the rest.

### Assumption

Consecutive frames are usually very similar.

### Frame Types

- **I-frame:** Intra frame. No change. No compression. Gives information for other frames.
- **P-frame:** Predicted frame. Predicted from previous frame. Example: phone at time t, next second it moves right. You can infer.
- **B-frame:** Bidirectional frame. Use frame at t and t+2 to infer t+1. Like interpolation.

### Limitations

- P-frame cannot handle fast motion or live streaming well.
- B-frame needs future frames. Live streaming/video call has no T+2. So it cannot be used for real-time.
- Sports video: too fast, hard to predict.
- Real-time video call: no future frames.

### Simple Example

- I-frame = key drawing in animation.
- P-frame = “same as before, but arm moves slightly.”
- B-frame = “between frame 1 and frame 3, draw frame 2.”

### Key Takeaway

> I = independent.  
> P = past.  
> B = both sides.  
> Live streaming hates B.

---

## 10. Image Restoration and Medical Imaging

### Space Imaging

The Ranger 7 probe took a picture of the moon minutes before landing. It was used for image restoration.

### Medical Imaging

MRI and X-ray devices have vibrations. Patients move or breathe. This disturbs data collection. Should we repeat the process? AI can restore the image.

### MRI Startup Example

- Advanced MRI is fast but very costly.
- Medium-range MRI takes 40–45 minutes.
- A startup designed a lower-cost MRI. Collected data is degraded.
- AI converts degraded image to high-end MRI image = **image-to-image translation**.
- Saves 25–30 minutes.

### Validation

If AI inference is not actual data, how do we validate?

- Train/validation/test sets.
- Medical benchmarks.
- Reinforcement learning: radiologist checks and gives feedback.

### Generalization

Learn mapping from low-resolution to high-resolution images. Then infer for new incoming images. Inference happens at deployment.

### Simple Example

Old photo restoration app. You give a blurry, scratched photo. AI makes it sharp. It learned from many before/after pairs.

### Key Takeaway

> Image restoration = learn low-to-high mapping, then infer at deployment.

---

## 11. How to Learn Computer Vision

- There is no single textbook that follows everything step by step.
- Slides are your main guide.
- Use GPT, Gemini, and other tools to understand deeply.
- Focus on logic over syntax. Coding is secondary.
- Before solving a problem, find the current state of the art.
- Search Google Scholar, papers, blogs, and company work.
- Then find gaps.

### Example: Drone Swarm

YOLO is used for collision avoidance in drone swarms. Five features per image are used.

### Example: UAV Classification

A UAV at 30,000 ft and 20 km distance faces poor AQI, humidity, and visibility issues. First, find what has been done. Then decide what else can be done.

### Coming Next

Brightness, contrast, HDR, and data distribution.

### Key Takeaway

> Develop the way of thinking first.  
> Then code. Then optimize.

---

## Cheat Sheet

| Concept | Simple Memory |
|---|---|
| 3D reconstruction | Many 2D photos + camera angles → 3D model |
| ImageNet | Fei-Fei Li, 2004, 22k categories, ILSVRC 1000 classes |
| Interclass vs. intraclass | Dog vs. wolf; same dog different poses |
| AlexNet/ResNet/Inception/ViT | Architecture turning points |
| Semantic vs. instance segmentation | All chairs blue vs. each chair different color |
| CLIP | Image + text contrastive learning. Best text match wins |
| SAM | Segmentation, not classification |
| DALL-E | Text → image |
| RGB vs. CMYK | Light vs. ink. Additive vs. subtractive |
| Occlusion | Hidden object. Fix with finer resolution or multiple cameras |
| Compression | Pack suitcase. Keep essentials |
| PNG vs. JPG | Lossless vs. lossy |
| I/P/B frames | Keyframe, predicted from past, predicted from both sides |
| MRI restoration | Low-res MRI → AI → high-res MRI. Image-to-image translation |
| Study advice | Understand, don’t mug. Logic over syntax |

---

## Practice Exercises

1. Take 5 photos of an object from different angles. Try to identify matching features manually.
2. Explain semantic vs. instance segmentation using a street scene.
3. Write a short note on how CLIP would classify an image of a “golden retriever” without training on that class.
4. List three situations where B-frames cannot be used.
5. Design a compression strategy for a surveillance camera with limited bandwidth.
6. Find one recent paper on image restoration and summarize its validation method.

---