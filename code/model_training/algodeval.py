import os
import shutil
import random
from tqdm import tqdm

# Path to the merged dataset
merged_path = '/home/enrique/Desktop/VARIOS/garzIA/pebble_contours/data/datasets/pebbles_final_filtered'

# Percentage of images to move from train to val
val_percentage = 0.2

# Function to move a subset of files from train to val
def move_files_to_val(image_dir, label_dir, val_image_dir, val_label_dir, val_percentage):
    # List all files in the train directory
    images = [f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))]

    # Calculate the number of files to move
    num_val_images = int(len(images) * val_percentage)

    # Randomly select files to move
    val_images = random.sample(images, num_val_images)

    for image in tqdm(val_images, desc=f"Moving {val_percentage * 100}% of images from {image_dir} to {val_image_dir}"):
        # Move image
        shutil.move(os.path.join(image_dir, image), os.path.join(val_image_dir, image))
        
        # Move corresponding label
        label = image.replace('.jpg', '.txt').replace('.png', '.txt')  # Handle both jpg and png extensions
        shutil.move(os.path.join(label_dir, label), os.path.join(val_label_dir, label))

# Paths to train and val directories for images and labels
image_train_dir = os.path.join(merged_path, 'images/train')
label_train_dir = os.path.join(merged_path, 'labels/train')
image_val_dir = os.path.join(merged_path, 'images/val')
label_val_dir = os.path.join(merged_path, 'labels/val')

# Create val directories if they don't exist
os.makedirs(image_val_dir, exist_ok=True)
os.makedirs(label_val_dir, exist_ok=True)

# Move files from train to val
move_files_to_val(image_train_dir, label_train_dir, image_val_dir, label_val_dir, val_percentage)

print("20% of the images and labels have been moved from train to val.")

