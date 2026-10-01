import os
import shutil
import random

# ============================================================
# CHANGE THIS PATH
# Put the path of the folder containing your 4 Kaggle folders
# ============================================================

SOURCE_DIR = r"C:\Users\YourName\Downloads\ECG Dataset"

# Project dataset folder
OUTPUT_DIR = "dataset"

# Train/test split
TRAIN_RATIO = 0.80

# Reproducible random split
random.seed(42)


# ============================================================
# Find folders automatically
# ============================================================

def find_folder(keyword):
    for name in os.listdir(SOURCE_DIR):
        full_path = os.path.join(SOURCE_DIR, name)

        if os.path.isdir(full_path) and keyword.lower() in name.lower():
            return full_path

    return None


normal_folder = find_folder("Normal Person")
abnormal_heartbeat_folder = find_folder("abnormal")
history_mi_folder = find_folder("History")
mi_folder = find_folder("Myocardial Infarction")


# ============================================================
# Check folders
# ============================================================

print("\nFound folders:")

print("Normal:", normal_folder)
print("Abnormal Heartbeat:", abnormal_heartbeat_folder)
print("History MI:", history_mi_folder)
print("Myocardial Infarction:", mi_folder)

if not normal_folder:
    raise Exception("Normal folder not found!")

if not abnormal_heartbeat_folder:
    raise Exception("Abnormal Heartbeat folder not found!")

if not history_mi_folder:
    raise Exception("History of MI folder not found!")

if not mi_folder:
    raise Exception("Myocardial Infarction folder not found!")


# ============================================================
# Create output directories
# ============================================================

folders = [
    os.path.join(OUTPUT_DIR, "train", "normal"),
    os.path.join(OUTPUT_DIR, "train", "abnormal"),
    os.path.join(OUTPUT_DIR, "test", "normal"),
    os.path.join(OUTPUT_DIR, "test", "abnormal")
]

for folder in folders:
    os.makedirs(folder, exist_ok=True)


# ============================================================
# Get image files
# ============================================================

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")


def get_images(folder):
    images = []

    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.lower().endswith(IMAGE_EXTENSIONS):
                images.append(os.path.join(root, file))

    return images


# ============================================================
# Copy images
# ============================================================

def split_and_copy(images, class_name):

    random.shuffle(images)

    split_index = int(len(images) * TRAIN_RATIO)

    train_images = images[:split_index]
    test_images = images[split_index:]

    train_folder = os.path.join(
        OUTPUT_DIR,
        "train",
        class_name
    )

    test_folder = os.path.join(
        OUTPUT_DIR,
        "test",
        class_name
    )

    print(f"\n{class_name.upper()}")
    print("Total:", len(images))
    print("Train:", len(train_images))
    print("Test:", len(test_images))

    # Copy training images
    for index, image_path in enumerate(train_images):

        extension = os.path.splitext(image_path)[1]

        destination = os.path.join(
            train_folder,
            f"{class_name}_train_{index}{extension}"
        )

        shutil.copy2(image_path, destination)

    # Copy testing images
    for index, image_path in enumerate(test_images):

        extension = os.path.splitext(image_path)[1]

        destination = os.path.join(
            test_folder,
            f"{class_name}_test_{index}{extension}"
        )

        shutil.copy2(image_path, destination)


# ============================================================
# Prepare NORMAL
# ============================================================

normal_images = get_images(normal_folder)

split_and_copy(
    normal_images,
    "normal"
)


# ============================================================
# Prepare ABNORMAL
# ============================================================

abnormal_images = []

abnormal_images.extend(
    get_images(abnormal_heartbeat_folder)
)

abnormal_images.extend(
    get_images(history_mi_folder)
)

abnormal_images.extend(
    get_images(mi_folder)
)

split_and_copy(
    abnormal_images,
    "abnormal"
)


# ============================================================
# Finished
# ============================================================

print("\n========================================")
print("DATASET PREPARATION COMPLETED")
print("========================================")

print("\nYour dataset is now:")

print("""
dataset/
│
├── train/
│   ├── normal/
│   └── abnormal/
│
└── test/
    ├── normal/
    └── abnormal/
""")