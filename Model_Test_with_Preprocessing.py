import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import matplotlib.pyplot as plt

# Path to the trained model
MODEL_PATH = "Crop Disease Prediction Model_CNN.h5"

# Load the trained model
model = load_model(MODEL_PATH)

# Define the classes (update these based on your trained model's classes)
DISEASE_CLASSES = ['Black Gram_Anthracnose', 'Black Gram_Healthy', 'Black Gram_Leaf Crinckle','Black Gram_Powdery Mildew','Black Gram_Yellow Mosaic','Rice_Bacterial Blight','Rice_BrownSpot','Rice_False Smut','Rice_Healthy','Rice_LeafBlast','Rice_Sheath Blight','Rice_Tungro']  # Replace with actual class names

# Function to preprocess the image
def preprocess_image(image_path, target_size=(224, 224)):
    """
    Preprocesses an image for prediction.
    Args:
        image_path (str): Path to the input image.
        target_size (tuple): Target size to resize the image (default: (224, 224)).
Returns:
        numpy.ndarray: Preprocessed image ready for model input.
    """
    img = load_img(image_path, target_size=target_size)  # Load and resize image
    img_array = img_to_array(img)  # Convert image to numpy array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize pixel values to [0, 1]
    return img_array

# Function to predict the disease
def predict_disease(image_path):
    """
    Predicts the class of the uploaded image.
    Args:
        image_path (str): Path to the input image.
    Returns:
        tuple: Predicted class and confidence score.
    """
    img_array = preprocess_image(image_path)
    predictions = model.predict(img_array)
    predicted_class = DISEASE_CLASSES[np.argmax(predictions)]  # Class with the highest probability
    confidence = np.max(predictions)  # Confidence score
    return predicted_class, confidence
# Upload and test the model
if __name__ == "__main__":

    # Path to the image for testing
    test_image_path = '1p.jpg'  # Example: 'path_to_image.jpg'

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
        print(f"Error: {e}")