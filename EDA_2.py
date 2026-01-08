# %%
# Import Libraries

import os
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from glob import glob
import cv2  # For image processing tasks like resizing
from sklearn.preprocessing import LabelEncoder


# %%
# Set Dataset Path*
# Define the path to the dataset folder (Root folder where class subfolders are stored)

dataset = tf.keras.preprocessing.image_dataset_from_directory("Dataset2")

# %%
# List the subfolders (disease categories)

class_folders = [f.name for f in os.scandir("Dataset2") if f.is_dir()]
print(f"Classes (diseases): {class_folders}")


# %%
# List All Images in Dataset*
# List all image paths across all disease categories for further exploration.

image_paths = glob(os.path.join("Dataset2", '*', '*.jpg'))  # Adjust for file type if needed
print(f"Total images: {len(image_paths)}")


# %%
# Display the first few image paths

print(image_paths[:5])


# %%
# Class Distribution
# Count the number of images in each class (folder)

class_counts = {class_name: len(glob(os.path.join("Dataset2", class_name, '*.jpg'))) for class_name in class_folders}
print(class_counts)


# %%
# Plot the distribution of images across classes

plt.figure(figsize=(10, 6))
sns.barplot(x=list(class_counts.keys()), y=list(class_counts.values()), palette='viridis', hue=list(class_counts.keys()), legend=False)
plt.title("Class Distribution: Crop Diseases")
plt.xlabel("Disease")
plt.ylabel("Number of Images")
plt.xticks(rotation=45)
plt.show()


# %%
# Check Image Size and Resolution
# Function to get image dimensions and size

def get_image_info(image_path):
    img = Image.open(image_path)
    return img.size  # (width, height)


# %%
# Get image dimensions for the first 5 images

image_info = [get_image_info(img_path) for img_path in image_paths[:5]]
print(image_info)


# %%
# Check if all images have the same resolution

unique_dimensions = set(image_info)
print(f"Unique image dimensions: {unique_dimensions}")


# %%
# Visualize Sample Images
# Function to display images

def show_image(image_path):
    img = Image.open(image_path)
    plt.imshow(img)
    plt.axis('off')
    plt.show()


# %%
# Display sample images (first 5 images)

for img_path in image_paths[:5]:
    show_image(img_path)


# %%
# Image Quality Check
# Check image quality (resolution, file size)

def check_image_quality(image_path):
    try:
        img = Image.open(image_path)
        return img.size, os.path.getsize(image_path)  
    
    # Returns resolution and file size
    except Exception as e:
        return None, None  # For corrupted images


# %%
# Check for the first few images

image_quality = [check_image_quality(img_path) for img_path in image_paths[:5]]
print(image_quality)

# %%
# Analyze the resolution and file size

resolutions = [info[0] for info in image_quality if info[0] is not None]
file_sizes = [info[1] for info in image_quality if info[1] is not None]

print(f"Resolution statistics: {np.unique(resolutions)}")
print(f"File size statistics: {np.min(file_sizes)} - {np.max(file_sizes)} bytes")

# %%
# Image Color Distribution (Optional)

# Plot the RGB histogram for a sample image
def plot_rgb_histogram(image_path):
    # Read the image
    img = cv2.imread(image_path)  # Open image with OpenCV in BGR format
    if img is None:
        print(f"Error: Unable to load image from {image_path}")
        return
    
    # Convert BGR to RGB
    r, g, b = cv2.split(img)  # Split into red, green, and blue channels
    
    # Plot the histogram for each channel
    plt.figure(figsize=(10, 6))
    plt.hist(r.ravel(), bins=256, color='red', alpha=0.5, label='Red')
    plt.hist(g.ravel(), bins=256, color='green', alpha=0.5, label='Green')
    plt.hist(b.ravel(), bins=256, color='blue', alpha=0.5, label='Blue')
    plt.title(f"RGB Histogram: {os.path.basename(image_path)}")
    plt.legend(loc='best')
    plt.show()


# %%
# Show RGB histogram for the first image

plot_rgb_histogram(image_paths[0])


# %%
# Check for Class Imbalance
# Check for class imbalance (normalized)

total_images = len(image_paths)
class_percentages = {class_name: (count / total_images) * 100 for class_name, count in class_counts.items()}

# %%
# Plot class distribution percentage

plt.figure(figsize=(10, 6))
sns.barplot(x=list(class_percentages.keys()), y=list(class_percentages.values()), color='skyblue')
plt.title("Class Distribution Percentage: Crop Diseases")
plt.xlabel("Disease")
plt.ylabel("Percentage of Total Images")
plt.xticks(rotation=45)
plt.show()


# %%
# Resize Images (if needed)
# Resize images to a standard target size (e.g., 224x224 for a CNN model)

target_size = (224, 224)

def resize_image(image_path, target_size):
    img = Image.open(image_path)
    img = img.resize(target_size)
    return np.array(img)


# %%
# Resize the first image and show it

resized_img = resize_image(image_paths[0], target_size)
plt.imshow(resized_img)
plt.title("Resized Image")
plt.axis('off')
plt.show()

# %%
# import os
# from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img, save_img
# import numpy as np

# # Define the base dataset directory and augmentation output directory
# dataset_dir = "Datasets"  # Replace with your dataset folder
# output_dir = "Datasets"  # Replace with your output folder
# os.makedirs(output_dir, exist_ok=True)

# # List of classes to augment
# classes_to_augment = ['Black Gram_Anthracnose', 'Black Gram_Healthy', 'Black Gram_Powdery Mildew', 'Black Gram_Yellow Mosaic']  # Replace with your class names
# # Augmentation configuration
# datagen = ImageDataGenerator(
#     rotation_range=30,  # Rotate images up to 30 degrees
#     width_shift_range=0.2,  # Horizontal shift
#     height_shift_range=0.2,  # Vertical shift
#     shear_range=0.2,  # Shearing transformation
#     zoom_range=0.2,  # Zoom in/out
#     horizontal_flip=True,  # Randomly flip images horizontally
#     fill_mode='nearest'  # Filling pixels after transformation
# )

# # Number of augmented images per original image
# num_augmented_images = 1

# # Augment images for selected classes
# for class_name in classes_to_augment:
#     class_path = os.path.join(dataset_dir, class_name)
#     output_class_path = os.path.join(output_dir, class_name)
#     os.makedirs(output_class_path, exist_ok=True)
    
#     if not os.path.exists(class_path):
#         print(f"Class folder '{class_name}' does not exist. Skipping.")
#         continue
    
#     for image_name in os.listdir(class_path):
#         image_path = os.path.join(class_path, image_name)
        
#         try:
#             img = load_img(image_path)  # Load the image
#             img_array = img_to_array(img)  # Convert to array
#             img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
            
#             # Generate augmented images
#             i = 0
#             for batch in datagen.flow(img_array, batch_size=1, save_to_dir=output_class_path,
#                                       save_prefix='aug', save_format='jpg'):
#                 i += 1
#                 if i >= num_augmented_images:
#                     break  # Stop after creating the specified number of augmented images
            
#         except Exception as e:
#             print(f"Error processing image {image_name}: {e}")
    
#     print(f"Augmentation completed for class '{class_name}'.")

# print("Data augmentation completed.")


# %%
# class_counts = {class_name: len(glob(os.path.join("Datasets", class_name, '*.jpg'))) for class_name in class_folders}
# print(class_counts)

# %%
# # Plot the distribution of images across classes

# plt.figure(figsize=(10, 6))
# sns.barplot(x=list(class_counts.keys()), y=list(class_counts.values()), palette='viridis', hue=list(class_counts.keys()), legend=False)
# plt.title("Class Distribution: Crop Diseases")
# plt.xlabel("Disease")
# plt.ylabel("Number of Images")
# plt.xticks(rotation=45)
# plt.show()


# %%
# import os
# from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img, save_img
# import numpy as np

# # Define the base dataset directory and augmentation output directory
# dataset_dir = "Datasets"  # Replace with your dataset folder
# output_dir = "Datasets"  # Replace with your output folder
# os.makedirs(output_dir, exist_ok=True)

# # List of classes to augment
# classes_to_augment = ['Rice_False Smut']  # Replace with your class names
# # Augmentation configuration
# datagen = ImageDataGenerator(
#     rotation_range=30,  # Rotate images up to 30 degrees
#     width_shift_range=0.2,  # Horizontal shift
#     height_shift_range=0.2,  # Vertical shift
#     shear_range=0.2,  # Shearing transformation
#     zoom_range=0.2,  # Zoom in/out
#     horizontal_flip=True,  # Randomly flip images horizontally
#     fill_mode='nearest'  # Filling pixels after transformation
# )

# # Number of augmented images per original image
# num_augmented_images = 5

# # Augment images for selected classes
# for class_name in classes_to_augment:
#     class_path = os.path.join(dataset_dir, class_name)
#     output_class_path = os.path.join(output_dir, class_name)
#     os.makedirs(output_class_path, exist_ok=True)
    
#     if not os.path.exists(class_path):
#         print(f"Class folder '{class_name}' does not exist. Skipping.")
#         continue
    
#     for image_name in os.listdir(class_path):
#         image_path = os.path.join(class_path, image_name)
        
#         try:
#             img = load_img(image_path)  # Load the image
#             img_array = img_to_array(img)  # Convert to array
#             img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
            
#             # Generate augmented images
#             i = 0
#             for batch in datagen.flow(img_array, batch_size=1, save_to_dir=output_class_path,
#                                       save_prefix='aug', save_format='jpg'):
#                 i += 1
#                 if i >= num_augmented_images:
#                     break  # Stop after creating the specified number of augmented images
            
#         except Exception as e:
#             print(f"Error processing image {image_name}: {e}")
    
#     print(f"Augmentation completed for class '{class_name}'.")

# print("Data augmentation completed.")


# %%
# import os
# from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img, save_img
# import numpy as np

# # Define the base dataset directory and augmentation output directory
# dataset_dir = "Datasets"  # Replace with your dataset folder
# output_dir = "Datasets"  # Replace with your output folder
# os.makedirs(output_dir, exist_ok=True)

# # List of classes to augment
# classes_to_augment = ['Black Gram_Leaf Crinckle']  # Replace with your class names
# # Augmentation configuration
# datagen = ImageDataGenerator(
#     rotation_range=30,  # Rotate images up to 30 degrees
#     width_shift_range=0.2,  # Horizontal shift
#     height_shift_range=0.2,  # Vertical shift
#     shear_range=0.2,  # Shearing transformation
#     zoom_range=0.2,  # Zoom in/out
#     horizontal_flip=True,  # Randomly flip images horizontally
#     fill_mode='nearest'  # Filling pixels after transformation
# )

# # Number of augmented images per original image
# num_augmented_images = 2

# # Augment images for selected classes
# for class_name in classes_to_augment:
#     class_path = os.path.join(dataset_dir, class_name)
#     output_class_path = os.path.join(output_dir, class_name)
#     os.makedirs(output_class_path, exist_ok=True)
    
#     if not os.path.exists(class_path):
#         print(f"Class folder '{class_name}' does not exist. Skipping.")
#         continue
    
#     for image_name in os.listdir(class_path):
#         image_path = os.path.join(class_path, image_name)
        
#         try:
#             img = load_img(image_path)  # Load the image
#             img_array = img_to_array(img)  # Convert to array
#             img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
            
#             # Generate augmented images
#             i = 0
#             for batch in datagen.flow(img_array, batch_size=1, save_to_dir=output_class_path,
#                                       save_prefix='aug', save_format='jpg'):
#                 i += 1
#                 if i >= num_augmented_images:
#                     break  # Stop after creating the specified number of augmented images
            
#         except Exception as e:
#             print(f"Error processing image {image_name}: {e}")
    
#     print(f"Augmentation completed for class '{class_name}'.")

# print("Data augmentation completed.")


# %%
# Assuming image filenames or folder names are disease labels
labels = [os.path.basename(os.path.dirname(img_path)) for img_path in image_paths]

# Encode labels
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)

# Show some encoded labels
print(f"Original Labels: {labels[:5]}")
print(f"Encoded Labels: {encoded_labels[:5]}")


# %%
#Save Metadata
# You may want to save metadata such as class labels, file paths, and encoded labels for easy access during training.
# Save as a CSV File

import pandas as pd

# Create a DataFrame with file paths and labels
labels = [os.path.basename(os.path.dirname(img_path)) for img_path in image_paths]
data = pd.DataFrame({'file_path': image_paths, 'label': labels})

# Save to CSV
metadata_path = "metadata_2.csv"
data.to_csv(metadata_path, index=False)
print(f"Metadata saved to {metadata_path}")


# %%
# Save Encoded Labels (if applicable)
# If you encoded the labels, save the mapping for reuse during model inference.

import pickle

# Save the label encoder
label_encoder_path = "label_encoder_2.pkl"
with open(label_encoder_path, 'wb') as file:
    pickle.dump(label_encoder, file)

print(f"Label encoder saved to {label_encoder_path}")


# %%
import numpy as np
from PIL import Image

# Example: Target size for all images
target_size = (224, 224)

resized_images = []
encoded_labels = []

for img_path in image_paths:
    img = Image.open(img_path)
    
    # Convert image to RGB to ensure consistent channels
    img = img.convert('RGB')
    
    # Resize image
    img = img.resize(target_size)
    
    # Append the processed image and its label
    resized_images.append(np.array(img))
    encoded_labels.append(label_encoder.transform([os.path.basename(os.path.dirname(img_path))])[0])

# Convert lists to NumPy arrays
resized_images = np.array(resized_images)  # Shape: (num_samples, 224, 224, 3)
encoded_labels = np.array(encoded_labels)

# Save the arrays to disk
np.save("resized_images.npy", resized_images)
np.save("encoded_labels.npy", encoded_labels)

print("Resized images and encoded labels saved successfully!")


# %%
#### Save as TFRecord (for TensorFlow)

import tensorflow as tf

# Define a function to create a TFRecord example
def create_tfrecord_example(image, label):
    feature = {
        'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.io.encode_jpeg(image).numpy()])),
        'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))
# Save as TFRecord
tfrecord_path = "crop_disease_data_2.tfrecord"
with tf.io.TFRecordWriter(tfrecord_path) as writer:
    for img, label in zip(resized_images, encoded_labels):
        example = create_tfrecord_example(img, label)
        writer.write(example.SerializeToString())

print(f"Dataset saved as TFRecord at {tfrecord_path}")
# %%
