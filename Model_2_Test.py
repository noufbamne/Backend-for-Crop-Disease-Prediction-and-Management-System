import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img
import numpy as np
import matplotlib.pyplot as plt

# Path to the trained model
MODEL_PATH = "Model_2.h5"

# Load the trained model
model = load_model(MODEL_PATH)

# Define the classes (update these based on your trained model's classes)
DISEASE_CLASSES = [
    ['Black Gram_Anthracnose',
     'Black Gram_Healthy',
     'Black Gram_Leaf Crinckle',
     'Black Gram_Powdery Mildew',
     'Rice_Bacterial Blight',
     'Unknown']
]  # Replace with actual class names

# Function to predict the disease without preprocessing
def predict_disease(image_path):
    """
    Predicts the class of the uploaded image.
    Args:
        image_path (str): Path to the input image.
    Returns:
        tuple: Predicted class and confidence score.
    """
    # Load the image as is
    img = load_img(image_path)  # Load image without any preprocessing
    img_array = np.array(img)  # Convert the image to a numpy array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Predict the class
    predictions = model.predict(img_array)
    predicted_class = DISEASE_CLASSES[np.argmax(predictions)]  # Class with the highest probability
    confidence = np.max(predictions)  # Confidence score
    return predicted_class, confidence

# Upload and test the model
if __name__ == "__main__":

    # Path to the image for testing
    test_image_path = '1a.jpg'  # Example: 'path_to_image.jpg'

    try:
        # Predict the disease
        predicted_class, confidence = predict_disease(test_image_path)

        # Display the image
        img = load_img(test_image_path)
        plt.imshow(img)
        plt.axis("off")
        plt.title(f"Predicted: {predicted_class} ({confidence * 100:.2f}%)")
        plt.show()

        print(f"Predicted Class: {predicted_class}, Confidence: {confidence * 100:.2f}%")
    except Exception as e:
        print(f"Error: {e}")
