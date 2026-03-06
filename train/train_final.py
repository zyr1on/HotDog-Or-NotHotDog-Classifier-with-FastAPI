import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import matplotlib.pyplot as plt
import numpy as np

# Seed settings for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Parameters
IMG_SIZE = 224  # MobileNetV2 input size
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.0001

# Data paths
TRAIN_DIR = 'hot-dog-not-hot-dog/train'
TEST_DIR = 'hot-dog-not-hot-dog/test'

print("=" * 60)
print("[INFO] HOT DOG CLASSIFIER - TRANSFER LEARNING")
print("=" * 60)

# Data Augmentation for training
train_datagen = ImageDataGenerator(
    rescale=1./255,  # Normalize pixel values to [0,1]
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2  # Reserve 20% for validation
)

# Test data generator (only normalization)
test_datagen = ImageDataGenerator(rescale=1./255)

# Data loaders
print("\n[INFO] Loading data...")
train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='training',
    shuffle=True
)

validation_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation',
    shuffle=True
)

test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False
)

print(f"\n[+] Training samples: {train_generator.samples}")
print(f"[+] Validation samples: {validation_generator.samples}")
print(f"[+] Test samples: {test_generator.samples}")
print(f"\n[INFO] Class Indices: {train_generator.class_indices}")
print(f"   [!]  NOTE: Alphabetical order -> 'hot_dog'=0, 'not_hot_dog'=1")
print(f"   [!]  Predictions: prediction > 0.5 = not_hot_dog, < 0.5 = hot_dog")

# Build model
print("\n[INFO] Building model...")

# Load pre-trained MobileNetV2 (ImageNet weights)
base_model = MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,  # Exclude the top layers
    weights='imagenet'
)

# Freeze base model initially
base_model.trainable = False

# Create custom model on top of base
model = keras.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(128, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(1, activation='sigmoid')  # Binary classification
])

# Compile the model
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss='binary_crossentropy',
    metrics=['accuracy', 
             keras.metrics.Precision(name='precision'),
             keras.metrics.Recall(name='recall')]
)

print("\n[INFO] Model Summary:")
model.summary()

# Callbacks
callbacks = [
    # Save the best model
    ModelCheckpoint(
        'best_hotdog_model.keras',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    ),
    
    # Early stopping to prevent overfitting
    EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    ),
    
    # Reduce learning rate on plateau
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        verbose=1
    )
]

# Training - Phase 1: Transfer Learning
print("\n" + "=" * 60)
print("[INFO] PHASE 1: TRANSFER LEARNING (Base model frozen)")
print("=" * 60)

history_phase1 = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=20,
    callbacks=callbacks,
    verbose=1
)

# Training - Phase 2: Fine-tuning
print("\n" + "=" * 60)
print("[INFO] PHASE 2: FINE-TUNING (Unfreezing top layers)")
print("=" * 60)

# Unfreeze the top layers of the base model
base_model.trainable = True

# Only train the last 30 layers
for layer in base_model.layers[:-30]:
    layer.trainable = False

# Recompile model with lower learning rate for fine-tuning
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE/10),
    loss='binary_crossentropy',
    metrics=['accuracy',
             keras.metrics.Precision(name='precision'),
             keras.metrics.Recall(name='recall')]
)

# Fine-tuning training
history_phase2 = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=EPOCHS,
    initial_epoch=len(history_phase1.history['loss']),
    callbacks=callbacks,
    verbose=1
)

# Combine histories
history = {
    'accuracy': history_phase1.history['accuracy'] + history_phase2.history['accuracy'],
    'val_accuracy': history_phase1.history['val_accuracy'] + history_phase2.history['val_accuracy'],
    'loss': history_phase1.history['loss'] + history_phase2.history['loss'],
    'val_loss': history_phase1.history['val_loss'] + history_phase2.history['val_loss']
}

# Evaluate on test set
print("\n" + "=" * 60)
print("[INFO] TEST SET EVALUATION")
print("=" * 60)

test_loss, test_accuracy, test_precision, test_recall = model.evaluate(test_generator)
f1_score = 2 * (test_precision * test_recall) / (test_precision + test_recall)

print(f"\n[+] Test Accuracy: {test_accuracy*100:.2f}%")
print(f"[+] Test Loss: {test_loss:.4f}")
print(f"[+] Precision: {test_precision*100:.2f}%")
print(f"[+] Recall: {test_recall*100:.2f}%")
print(f"[+] F1-Score: {f1_score*100:.2f}%")


# Save the final model
model.save('hotdog_classifier_final.keras')
print(f"\n[INFO] Model saved: hotdog_classifier_final.keras")

# Plot training graphs
print("\n[INFO] Generating training graphs...")

plt.figure(figsize=(15, 5))

# Accuracy plot
plt.subplot(1, 2, 1)
plt.plot(history['accuracy'], label='Train Accuracy', linewidth=2)
plt.plot(history['val_accuracy'], label='Validation Accuracy', linewidth=2)
plt.axvline(x=20, color='r', linestyle='--', label='Fine-tuning starts')
plt.title('Model Accuracy', fontsize=14, fontweight='bold')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True, alpha=0.3)

# Loss plot
plt.subplot(1, 2, 2)
plt.plot(history['loss'], label='Train Loss', linewidth=2)
plt.plot(history['val_loss'], label='Validation Loss', linewidth=2)
plt.axvline(x=20, color='r', linestyle='--', label='Fine-tuning starts')
plt.title('Model Loss', fontsize=14, fontweight='bold')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
print("[INFO] Training graphs saved: training_history.png")

# Example predictions
print("\n" + "=" * 60)
print("SAMPLE PREDICTIONS")
print("=" * 60)

test_generator.reset()
predictions = model.predict(test_generator, verbose=1)
predicted_classes = (predictions > 0.5).astype(int).flatten()
true_classes = test_generator.classes

# Confusion matrix
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns


cm = confusion_matrix(true_classes, predicted_classes)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['hot_dog', 'not_hot_dog'],
            yticklabels=['hot_dog', 'not_hot_dog'])
plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
print("[+] Confusion matrix saved: confusion_matrix.png")

# Detailed classification report
print("\n[INFO] Detailed Classification Report:")
print(classification_report(true_classes, predicted_classes, 
                          target_names=['hot_dog', 'not_hot_dog']))

print("\n" + "=" * 60)
print("[!] TRAINING COMPLETED!")
print("=" * 60)

print(f"\n[+] Saved files:")
print(f"   • best_hotdog_model.keras (Best model based on validation accuracy)")
print(f"   • hotdog_classifier_final.keras (Final model after fine-tuning)")
print(f"   • training_history.png (Training accuracy & loss graphs)")
print(f"   • confusion_matrix.png (Confusion matrix)")