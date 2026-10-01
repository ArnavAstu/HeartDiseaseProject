# ============================================================
# IMPORT REQUIRED LIBRARIES
# ============================================================

import os
import json

# NumPy is used for numerical operations and prediction arrays
import numpy as np

# Matplotlib is used to generate training graphs
import matplotlib.pyplot as plt


# ImageDataGenerator is used to load images from folders,
# normalize images and perform data augmentation
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# MobileNetV2 is our pre-trained deep learning model
from tensorflow.keras.applications import MobileNetV2

# These are the additional layers that we add on top
# of the MobileNetV2 base model
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    GlobalAveragePooling2D
)

# Model is used to create the final neural network
from tensorflow.keras.models import Model

# Adam is the optimization algorithm used during training
from tensorflow.keras.optimizers import Adam

# Callbacks help control training and save the best model
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint
)


# Scikit-learn provides evaluation metrics
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)


# ============================================================
# CONFIGURATION
# ============================================================

# MobileNetV2 expects images of size 224 x 224
IMG_SIZE = 224

# Number of images processed in one training step
BATCH_SIZE = 16

# Maximum number of times the complete training data
# can be passed through the model
EPOCHS = 15


# Location of training images
TRAIN_DIR = "dataset/train"

# Location of testing images
TEST_DIR = "dataset/test"


# Folder where the trained model will be stored
MODEL_DIR = "model"

# Folder where graphs and evaluation results will be stored
RESULTS_DIR = "results"


# Create the model and results folders if they don't exist
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================================
# DATA AUGMENTATION AND PREPROCESSING
# ============================================================

# ImageDataGenerator prepares the training images.
#
# rescale:
# Converts pixel values from 0-255 to approximately 0-1.
#
# rotation_range:
# Allows small rotations of up to 5 degrees.
#
# width_shift_range / height_shift_range:
# Slightly shifts images horizontally and vertically.
#
# zoom_range:
# Slightly zooms images.
#
# horizontal_flip=False:
# We do NOT horizontally flip ECG images.
#
# validation_split=0.20:
# 20% of the training directory is reserved for validation.

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,

    rotation_range=5,

    width_shift_range=0.05,

    height_shift_range=0.05,

    zoom_range=0.10,

    horizontal_flip=False,

    validation_split=0.20
)


# Test images should NOT be augmented.
#
# We only normalize their pixel values.

test_datagen = ImageDataGenerator(
    rescale=1.0 / 255
)


# ============================================================
# TRAINING DATA
# ============================================================

# flow_from_directory() reads images from:
#
# dataset/train/
# ├── abnormal/
# └── normal/
#
# target_size:
# Resizes every image to 224 x 224.
#
# batch_size:
# Processes 16 images at a time.
#
# class_mode="binary":
# Tells Keras that this is a binary classification problem.
#
# subset="training":
# Uses the 80% training portion.
#
# shuffle=True:
# Randomizes the training images.

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,

    target_size=(IMG_SIZE, IMG_SIZE),

    batch_size=BATCH_SIZE,

    class_mode="binary",

    subset="training",

    shuffle=True
)


# ============================================================
# VALIDATION DATA
# ============================================================

# This generator uses the remaining 20% of the training data
# for validation.
#
# Validation data helps us monitor how well the model performs
# on images that were not directly used for updating weights.

validation_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,

    target_size=(IMG_SIZE, IMG_SIZE),

    batch_size=BATCH_SIZE,

    class_mode="binary",

    subset="validation",

    shuffle=False
)


# ============================================================
# TEST DATA
# ============================================================

# The test dataset is completely separate from training.
#
# It is used after training to evaluate the final model.

test_generator = test_datagen.flow_from_directory(
    TEST_DIR,

    target_size=(IMG_SIZE, IMG_SIZE),

    batch_size=BATCH_SIZE,

    class_mode="binary",

    shuffle=False
)


# ============================================================
# DISPLAY CLASS MAPPING
# ============================================================

# This tells us which numerical label Keras assigned
# to each folder.
#
# IMPORTANT:
# With folders named "abnormal" and "normal", Keras normally
# sorts them alphabetically:
#
# abnormal -> 0
# normal   -> 1
#
# So ALWAYS check the output of this print statement.

print("\nClass mapping:")
print(train_generator.class_indices)


# ============================================================
# MOBILE NET V2 BASE MODEL
# ============================================================

# Load MobileNetV2.
#
# weights="imagenet":
# Uses weights learned from the ImageNet dataset.
#
# include_top=False:
# Removes MobileNetV2's original classification layer.
#
# input_shape:
# Our images are 224 x 224 RGB images.

base_model = MobileNetV2(
    weights="imagenet",

    include_top=False,

    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)


# Freeze the pre-trained MobileNetV2 layers.
#
# This means their weights will not be updated during
# our initial training.
#
# Therefore MobileNetV2 mainly acts as a feature extractor.

base_model.trainable = False


# ============================================================
# ADD OUR CUSTOM CLASSIFICATION LAYERS
# ============================================================

# Get the output features generated by MobileNetV2.

x = base_model.output


# Global Average Pooling converts the feature maps
# into a compact feature vector.

x = GlobalAveragePooling2D()(x)


# Add a fully connected layer with 128 neurons.
#
# ReLU is used as the activation function.

x = Dense(
    128,
    activation="relu"
)(x)


# Dropout randomly disables 40% of neurons during training.
#
# This helps reduce overfitting.

x = Dropout(0.4)(x)


# Final classification layer.
#
# Dense(1):
# Only one output neuron is required for binary classification.
#
# sigmoid:
# Produces a value between 0 and 1.

output = Dense(
    1,
    activation="sigmoid"
)(x)


# ============================================================
# CREATE FINAL MODEL
# ============================================================

# Combine:
#
# MobileNetV2 base
# +
# Global Average Pooling
# +
# Dense layer
# +
# Dropout
# +
# Sigmoid output

model = Model(
    inputs=base_model.input,

    outputs=output
)


# ============================================================
# COMPILE MODEL
# ============================================================

model.compile(

    # Adam adjusts the model's weights during training.
    optimizer=Adam(
        learning_rate=0.0001
    ),

    # Binary Cross Entropy is appropriate for binary
    # classification.
    loss="binary_crossentropy",

    # Accuracy is used as a training metric.
    metrics=["accuracy"]
)


# Print the complete neural network architecture
model.summary()


# ============================================================
# CALLBACKS
# ============================================================

# Location where the best model will be saved.

model_path = os.path.join(
    MODEL_DIR,
    "ecg_model.h5"
)


callbacks = [

    # --------------------------------------------------------
    # EARLY STOPPING
    # --------------------------------------------------------
    #
    # Monitor validation loss.
    #
    # If validation loss does not improve for 4 epochs,
    # training stops.
    #
    # restore_best_weights=True:
    # Restores the weights from the best validation result.

    EarlyStopping(
        monitor="val_loss",

        patience=4,

        restore_best_weights=True
    ),


    # --------------------------------------------------------
    # MODEL CHECKPOINT
    # --------------------------------------------------------
    #
    # Saves the model whenever validation accuracy improves.
    #
    # save_best_only=True:
    # Only the best-performing model is saved.

    ModelCheckpoint(
        model_path,

        monitor="val_accuracy",

        save_best_only=True
    )

]


# ============================================================
# TRAIN THE MODEL
# ============================================================

print("\nStarting training...\n")


# model.fit() performs the actual training.
#
# train_generator:
# Provides training images.
#
# validation_data:
# Provides validation images.
#
# epochs:
# Maximum 15 training epochs.
#
# callbacks:
# Early stopping + model checkpointing.

history = model.fit(

    train_generator,

    validation_data=validation_generator,

    epochs=EPOCHS,

    callbacks=callbacks
)


# ============================================================
# EVALUATE MODEL ON TEST DATA
# ============================================================

print(
    "\nEvaluating model on test dataset...\n"
)


# Evaluate the trained model using unseen test images.

test_loss, test_accuracy = model.evaluate(
    test_generator
)


# Convert accuracy from decimal to percentage.

print(
    f"Test Accuracy: {test_accuracy * 100:.2f}%"
)


# ============================================================
# GENERATE PREDICTIONS
# ============================================================

# Ask the trained model to predict every image
# in the test dataset.

probabilities = model.predict(
    test_generator
)


# Convert probabilities into class labels.
#
# A threshold of 0.5 is used.
#
# probability > 0.5 -> class 1
# probability <= 0.5 -> class 0

predictions = (
    probabilities > 0.5
).astype(int).flatten()


# Get the actual labels of the test images.

true_labels = test_generator.classes


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

# Human-readable class names.
#
# IMPORTANT:
# These names MUST match the numerical mapping generated
# by flow_from_directory().

class_names = [
    "NORMAL",
    "ABNORMAL"
]


print(
    "\nClassification Report:\n"
)


# Generate:
#
# Precision
# Recall
# F1-score
# Support

report = classification_report(

    true_labels,

    predictions,

    target_names=class_names
)


print(report)


# ============================================================
# CONFUSION MATRIX
# ============================================================

# Create a confusion matrix using:
#
# Actual labels
# vs
# Predicted labels

cm = confusion_matrix(

    true_labels,

    predictions
)


# Convert the confusion matrix into a display object.

display = ConfusionMatrixDisplay(

    confusion_matrix=cm,

    display_labels=class_names
)


# Display the confusion matrix.

display.plot()


plt.title(
    "ECG Classification Confusion Matrix"
)


# Save the confusion matrix image.

plt.savefig(

    os.path.join(
        RESULTS_DIR,
        "confusion_matrix.png"
    ),

    bbox_inches="tight"
)


# Close the figure to free memory.

plt.close()


# ============================================================
# TRAINING ACCURACY GRAPH
# ============================================================

plt.figure()


# Training accuracy over epochs.

plt.plot(

    history.history["accuracy"],

    label="Training Accuracy"
)


# Validation accuracy over epochs.

plt.plot(

    history.history["val_accuracy"],

    label="Validation Accuracy"
)


plt.title(
    "Training and Validation Accuracy"
)


plt.xlabel(
    "Epoch"
)


plt.ylabel(
    "Accuracy"
)


plt.legend()


# Save graph.

plt.savefig(

    os.path.join(
        RESULTS_DIR,
        "training_accuracy.png"
    ),

    bbox_inches="tight"
)


plt.close()


# ============================================================
# TRAINING LOSS GRAPH
# ============================================================

plt.figure()


# Training loss over epochs.

plt.plot(

    history.history["loss"],

    label="Training Loss"
)


# Validation loss over epochs.

plt.plot(

    history.history["val_loss"],

    label="Validation Loss"
)


plt.title(
    "Training and Validation Loss"
)


plt.xlabel(
    "Epoch"
)


plt.ylabel(
    "Loss"
)


plt.legend()


# Save graph.

plt.savefig(

    os.path.join(
        RESULTS_DIR,
        "training_loss.png"
    ),

    bbox_inches="tight"
)


plt.close()


# ============================================================
# SAVE CLASS NAMES
# ============================================================

# IMPORTANT:
#
# Your original code hardcodes:
#
# 0 -> NORMAL
# 1 -> ABNORMAL
#
# But flow_from_directory() with folders:
#
# abnormal/
# normal/
#
# normally produces:
#
# 0 -> abnormal
# 1 -> normal
#
# Therefore this section needs to be handled carefully.

class_mapping = {

    "0": "NORMAL",

    "1": "ABNORMAL"

}


# Save the mapping in JSON format.

with open(

    os.path.join(
        MODEL_DIR,
        "class_names.json"
    ),

    "w"

) as file:

    json.dump(

        class_mapping,

        file,

        indent=4
    )


# ============================================================
# FINAL MESSAGE
# ============================================================

print(
    "\n========================================"
)

print(
    "MODEL TRAINING COMPLETED"
)

print(
    "========================================"
)


print(
    "\nModel saved at:"
)


print(
    model_path
)


print(
    "\nResults saved inside:"
)


print(
    RESULTS_DIR
)