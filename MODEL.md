# Model Architecture and Training

## Overview

The classifier is a fine-tuned MobileNetV2 network trained for binary image classification: hotdog vs. not hotdog. Transfer learning is used to leverage ImageNet-pretrained weights, with a lightweight custom head for the binary output.

---

## Base Model

**MobileNetV2** is a lightweight convolutional neural network designed for mobile and embedded vision applications. It uses depthwise separable convolutions and inverted residual blocks with linear bottlenecks, which makes it computationally efficient without a significant loss in accuracy.

The base model is loaded with `include_top=False`, discarding the original ImageNet classification head and retaining only the feature extraction backbone.

---

## Full Architecture

```
Input (224 x 224 x 3)
        |
MobileNetV2 backbone (ImageNet weights)
        |
GlobalAveragePooling2D
        |
BatchNormalization
        |
Dropout(0.5)
        |
Dense(128, activation='relu')
        |
BatchNormalization
        |
Dropout(0.3)
        |
Dense(1, activation='sigmoid')
        |
Output: score in [0, 1]
```

The sigmoid output is thresholded at 0.5. Scores below 0.5 map to **HOT DOG**, scores at or above 0.5 map to **NOT HOT DOG**.

---

## Training Pipeline

Training is split into two sequential phases.

### Phase 1 — Transfer Learning

The MobileNetV2 backbone is fully frozen. Only the custom head layers are trained. This allows the head to converge quickly on top of stable, general-purpose features without corrupting the pretrained weights.

| Parameter | Value |
|---|---|
| Epochs | 20 |
| Learning rate | 1e-4 |
| Trainable layers | Custom head only |

### Phase 2 — Fine-tuning

The top 30 layers of MobileNetV2 are unfrozen and trained jointly with the head at a reduced learning rate. This allows the backbone to adapt its higher-level features to the hotdog domain.

| Parameter | Value |
|---|---|
| Epochs | up to 50 (early stopping applies) |
| Learning rate | 1e-5 |
| Trainable layers | Top 30 MobileNetV2 layers + full head |

---

## Data Augmentation

Applied only to the training set. The validation and test sets receive only rescaling.

| Augmentation | Value |
|---|---|
| Rescaling | 1/255 |
| Rotation | up to 20 degrees |
| Width / height shift | 0.2 |
| Shear | 0.2 |
| Zoom | 0.2 |
| Horizontal flip | enabled |
| Fill mode | nearest |

Validation split of 20% is taken from the training directory.

---

## Callbacks

| Callback | Monitors | Behavior |
|---|---|---|
| `ModelCheckpoint` | `val_accuracy` | Saves model when validation accuracy improves |
| `EarlyStopping` | `val_loss` | Stops training if no improvement for 10 epochs; restores best weights |
| `ReduceLROnPlateau` | `val_loss` | Halves learning rate after 5 stagnant epochs; floor at 1e-7 |

---

## Inference

At inference time, each image is:

1. Decoded from bytes using Pillow
2. Converted to RGB (handles grayscale and RGBA inputs)
3. Resized to 224 x 224
4. Normalized to [0, 1] by dividing by 255
5. Expanded to shape `(1, 224, 224, 3)` for batch dimension

The model then produces a single sigmoid score. The `HotdogPredictor` class in `model_helper.py` encapsulates this pipeline and is instantiated once at server startup to avoid repeated model loading.

---

## Class Index Note

Keras assigns class indices alphabetically when loading data with `flow_from_directory`. This produces:

```
hot_dog      → 0
not_hot_dog  → 1
```

The sigmoid output therefore represents the probability of the `not_hot_dog` class. The threshold logic is:

```python
label = "NOT HOT DOG" if prediction_score > 0.5 else "HOT DOG"
```

---

## Output Files

| File | Description |
|---|---|
| `best_hotdog_model.keras` | Checkpoint saved at peak validation accuracy |
| `hotdog_classifier_final.keras` | Model state at the end of training |
| `training_history.png` | Accuracy and loss curves across both phases |
| `confusion_matrix.png` | Confusion matrix evaluated on the test set |
