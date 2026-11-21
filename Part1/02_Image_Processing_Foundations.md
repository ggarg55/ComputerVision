# Image Processing Foundations for Computer Vision and AI

A practical, intuition-first guide to histograms, brightness, contrast, dynamic range, noise, and early vision tasks.

This tutorial is a standalone reference for anyone who wants to understand what happens to an image **before** it reaches a machine learning model. The goal is not to memorize formulas, but to build the conceptual thinking needed to design better vision systems.

---

## Table of Contents

1. Digital Images in Brief  
2. Histogram  
3. Brightness  
4. Contrast  
5. Dynamic Range  
6. Image Negative  
7. Histogram Equalization  
8. Dynamic Range Compression / Log Transform  
9. Intensity Slicing / Image Enhancement  
10. Noise  
11. From Pixels to Semantics  
12. Motion and Descriptors  
13. When Do We Need High/Low Brightness and Contrast?  
14. Summary Cheat Sheet  
15. Practice Exercises  

---

## 1. Digital Images in Brief

A grayscale image is a matrix of pixels.  
Each pixel has an intensity value.

For an 8-bit grayscale image:

- Values range from **0 to 255**.
- 0 = black
- 255 = white
- Total possible values = **256 distinct gray levels**

A pixel at location `(x, y)` has intensity `I(x, y)`.

Everything in this tutorial builds on this simple idea: an image is numbers, and those numbers have patterns.

---

## 2. Histogram

A histogram counts how many times each intensity value appears in an image.

It is a **bar chart** or **frequency distribution**:

- How many pixels have value 0?
- How many have value 1?
- …
- How many have value 255?

### Key idea

A histogram gives **statistical information only**.

It tells you **how many**, not **where**.

For example, three different 8×8 images can have the exact same histogram but look completely different:

- One image: top half black, bottom half white.
- Another image: checkerboard pattern.
- Another image: random noise.

All three can have the same counts of black and white pixels, but their spatial arrangements are different.

So:

> **Histogram = frequency of intensities.  
> It does not encode location.**

Histograms are still extremely useful for understanding exposure, contrast, and for preprocessing steps like equalization.

---

## 3. Brightness

Brightness is how light or dark an image appears.

For a grayscale image, brightness is the **average intensity of all pixels**.

\[
\text{Brightness} = \frac{1}{N} \sum I(x, y)
\]

where `N` is the total number of pixels.

### Human view vs computer view

- Human: increasing brightness makes white light stronger.
- Computer: increasing brightness increases pixel intensities on average.

### Histogram and exposure

A histogram shows how brightness values are distributed.

- **Underexposed**: sharp peak near low intensities. The image looks dark. Brightness is not fully utilized.
- **Well-exposed**: histogram is spread out. Good distribution.
- **Overexposed**: values bunched at one level or near the high end. The image may look washed out or flat.

So:

> **Brightness tells you where the histogram is located.**

---

## 4. Contrast

Contrast indicates how easily objects in an image can be distinguished.

- **High contrast**: many distinct intensity values. Edges and details are clear.
- **Low contrast**: very few intensity values. The image looks dull, smooth, or blurry.

Contrast depends on how much brightness varies compared to the average brightness.

Larger variation = higher contrast.

### Histogram and contrast

- Narrow central peak → low contrast.
- Widespread histogram → good contrast.
- Peaks at the ends → high contrast, but possible loss of details or noise.
- Too much contrast can make an image look noisy or disturbed.

So:

> **Contrast tells you how widely the histogram is spread.**

---

## 5. Dynamic Range

Dynamic range refers to the **number of distinct brightness levels** an image uses.

It shows how much tonal variation exists.

- **High dynamic range (HDR)**: many shades. Rich tonal variation.
- **Low dynamic range (LDR)**: fewer shades. Flat look.

Simple analogy:

- Brightness = average height of a staircase.
- Contrast = difference between highest and lowest step.
- Dynamic range = number of steps.

So:

> **Brightness = location.  
> Contrast = spread.  
> Dynamic range = number of distinct levels.**

---

## 6. Image Negative

The negative of an image is simple:

\[
s = 255 - r
\]

where `r` is the original pixel value and `s` is the new value.

Examples:

- 0 → 255
- 255 → 0
- 100 → 155

This is the same effect used in mobile camera filters for black-and-white negatives.

---

## 7. Histogram Equalization

Histogram equalization improves contrast by spreading crowded intensity values over a wider range.

It makes previously similar-looking regions easier to distinguish.

### Goal

- Enhance image contrast.
- Spread pixel intensities to use the full dynamic range.
- Make the histogram more uniform.

It does **not** guarantee a perfectly flat histogram. It tries to make better use of the available intensity range.

### Why it matters

Before equalization, a histogram may be concentrated in a narrow range. The image looks dull, washed out, and fine structures are missing.

After equalization, the histogram is spread more widely. Dark and bright regions become more distinguishable.

### Step-by-step example

Consider a 4×4, 8-bit grayscale image.

Values: 80, 100, 120, 140.  
Each appears 4 times.

| Value | Count | Probability | CDF |
|-------|-------|-------------|-----|
| 80    | 4     | 0.25        | 0.25 |
| 100   | 4     | 0.25        | 0.50 |
| 120   | 4     | 0.25        | 0.75 |
| 140   | 4     | 0.25        | 1.00 |

CDF minimum = 0.25.

Mapping formula:

\[
s_k = \frac{\text{CDF}(k) - \text{CDF}_{\min}}{1 - \text{CDF}_{\min}} \times 255
\]

Results:

- 80 → 0
- 100 → 85
- 120 → 170
- 140 → 255

So the original range 80–140 becomes 0–255.

Now:

- 80 becomes darker.
- 140 becomes white.
- 100 becomes dark gray.
- 120 becomes light gray.

Contrast improves because variation is larger.

### Python example

```python
import numpy as np

def histogram_equalization(img):
    # img: 2D numpy array of uint8
    hist, bins = np.histogram(img.flatten(), 256, [0, 256])
    cdf = hist.cumsum()
    cdf_m = np.ma.masked_equal(cdf, 0)
    cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
    cdf = np.ma.filled(cdf_m, 0).astype('uint8')
    return cdf[img]
```

### Benefits

- Improves contrast.
- Uses full dynamic range.
- Normalizes images from different sources or exposures.
- Helps AI models see features better.

### Limitations

- Can amplify noise.
- Not always visually desirable.
- May wash out some regions.

---

## 8. Dynamic Range Compression / Log Transform

Some images contain a very large range of intensity values.

A few pixels may be extremely bright while many others are relatively dark.

If displayed directly, the bright pixels dominate and details in darker regions become difficult to see.

### Log transform

\[
s = c \log(1 + r)
\]

where `r` is the input pixel value and `c` is a scaling constant.

The log function compresses larger values more than smaller ones.

Why?

- It gives more penalty to high values.
- It brings extreme values into a manageable range.
- It balances the distribution of pixels.

### Use cases

- Medical imaging
- Astronomical data
- Satellite and remote sensing
- Google Maps / weather imagery
- Any scene with extreme bright and dark regions

### Why before AI?

If you pass a very dark image with bright dominating regions directly to an AI model:

- The model may learn a bias toward dark or bright regions.
- Edges may not be identifiable.
- Feature extraction can fail.
- Fine details like tumors may be missed.

Preprocessing helps the model see what matters.

---

## 9. Intensity Slicing / Image Enhancement

Sometimes you want to highlight only a specific range of intensities.

For example, range `A` to `B`.

Pixels in that range remain bright.  
Most other pixels become dark.

This makes the desired structure much easier to observe.

### Applications

- Medical imaging: highlight tissues or abnormalities.
- Video understanding: focus on one object, reduce computation.
- Object detection: crop or isolate regions of interest.
- Helping AI models learn better by removing background dominance.

### How to select the region?

- Manually with bounding boxes.
- Automatically with object detection models.
- Later lectures in computer vision cover bounding box prediction and segmentation.

Difference from a dark image:

- A dark image still has variation and distribution.
- In intensity slicing, outside the selected range, pixels become completely dark. No pattern. The model won’t be dominated by background.

---

## 10. Noise

Noise is unwanted signal added to an image.

It comes from environmental conditions, weather, device limitations, and capture process.

### Common types

| Noise Type | Description | Common Filter |
|------------|-------------|---------------|
| Gaussian / White | Many small random changes. Example: 116 becomes 118. | Averaging filter |
| Impulse / Salt-and-Pepper | Few extreme changes. Pixels become 0 or 255. | Median filter |
| Speckle | Granular, multiplicative. Common in ultrasound, MRI. | Median / adaptive |
| Shot | Photon counting noise. Low-light conditions. | Domain-specific |

### Difference between salt-and-pepper and speckle

- Salt-and-pepper: sudden extreme black/white pixels.
- Speckle: gradual, multiplicative, granular changes.

### Filters

- **Averaging filter**: 3×3 kernel of `1/9`.
- **Median filter**: replace each pixel with median of neighbors.
- **Sharpening filter**: makes edges and fine details sharper.
- **Lowpass filtering**: smooths image.
- **Spatial domain vs frequency domain**: kernels on pixels vs frequency-level processing.

### Noise is also added intentionally

To test model robustness:

- Dataset + X noise
- Dataset + Y noise
- Dataset + Z noise
- Combinations like X + Y + Z

Denoising models learn from noisy data and become robust.

---

## 11. From Pixels to Semantics

Once preprocessing is done, we move to higher-level computer vision tasks.

### Image detection

- Classification + localization.
- What object is it?
- Where is it?
- Bounding box around object.

### Image segmentation

- Fine-grain, every pixel.
- Not just a box, but exact shape.

### Video processing

- 2D image + time / motion.
- Video = image + motion.

---

## 12. Motion and Descriptors

### Optical flow

For each pixel, optical flow gives:

- Direction
- Magnitude

Color in optical flow images indicates direction and speed.

- **Sparse optical flow**: only some objects or feature points.
- **Dense optical flow**: every pixel. High computation. Used for pedestrian detection, crowded scenes.

### Descriptors

| Descriptor | Meaning | What it captures |
|------------|---------|------------------|
| HOG | Histogram of Oriented Gradients | Shape and structure. High-level, not exact shape or color. |
| HOF | Histogram of Oriented Optical Flow | Motion direction. Direction-sensitive. |
| MBH | Motion Boundary Histogram | Where motion changes. |

Example:

Two people walking in opposite directions.

- Their HOF features will be different.
- Why? Direction bins are different.
- Velocity vectors are reversed.
- Peaks appear on opposite sides of the histogram.

If HOF ignored motion direction, they might look similar. But HOF is direction-sensitive.

---

## 13. When Do We Need High/Low Brightness and Contrast?

Do we always need high brightness and high contrast? No.

| Scenario | Brightness | Contrast | Examples |
|----------|------------|----------|----------|
| High | High | OCR, document scanning, industrial inspection, daylight surveillance |
| Low | Low | Mood photography, night scenes, minimizing disturbance, artistic effects |
| High | Low | Foggy/overcast scenes, background plates, some medical ultrasound, soft look |
| Low | High | Night cityscapes, astronomy, fluorescence microscopy, security at night with lights |

Understanding the goal helps you choose the right preprocessing.

---

## 14. Summary Cheat Sheet

| Concept | Meaning | Histogram Relation |
|---------|---------|-------------------|
| Histogram | Frequency of intensities | Counting only, no location |
| Brightness | Average intensity | Where the histogram is located |
| Contrast | Distinguishability of objects | How widely the histogram is spread |
| Dynamic range | Number of distinct levels | Number of bins used |
| Negative | `255 - r` | Inverts intensities |
| Equalization | Spread crowded intensities | Flattens / widens histogram |
| Log transform | Compress high values | Balances extreme dynamic range |
| Intensity slicing | Highlight range A–B | Keeps selected range bright |
| Noise | Unwanted signal | Corrupts histogram |
| HOG | Shape / structure | Gradient orientation |
| HOF | Motion direction | Optical flow orientation |
| MBH | Where motion changes | Motion boundaries |

---

## 15. Practice Exercises

1. Create a 4×4 matrix with values 0, 50, 100, 150. Compute its histogram.
2. Apply the negative transform to the matrix.
3. Perform histogram equalization on the same matrix.
4. Identify the noise type in a given image: Gaussian, salt-and-pepper, speckle, or shot.
5. Explain why two people walking in opposite directions have different HOF features.
6. List four situations where low contrast is preferred over high contrast.

---

## Final Thoughts

Conceptual understanding beats memorizing formulas.

Algorithms are necessary, but without conceptual thinking, you cannot design good algorithms.

The pipeline is:

**Image quality → preprocessing → features → model → task.**

Master the foundations, and you can debug any vision system like an architect.

---