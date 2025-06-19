import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np

# Function to preprocess the uploaded image
def preprocess_image(image, target_size=(128, 128)):  # Adjust target_size based on model.input_shape
    # Convert Streamlit uploaded file to PIL Image
    img = Image.open(image).convert('RGB')
    # Resize image
    img = img.resize(target_size)
    # Convert to array
    img_array = img_to_array(img)
    # Normalize pixel values (scale to [0, 1] as commonly expected)
    img_array = img_array / 255.0  # Adjust if model expects [-1, 1] or other
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Function to predict the class of the image
def predict_image(image, model, class_labels):
    # Preprocess image
    processed_image = preprocess_image(image)
    # Predict
    predictions = model.predict(processed_image)
    # Get predicted class and confidence
    predicted_class = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class]
    # Return class label and confidence
    return class_labels[predicted_class], confidence

# Main Streamlit app
def main():
    # Set page title and description
    st.title("Fresh and Stale Fruit/Vegetable Classifier")
    st.write("Upload an image of a fruit or vegetable, and the model will predict if it's fresh or stale. Supported items: Apple, Banana, BitterGourd, Capsicum, Orange, Tomato.")

    # Load model (update the path to your model file)
    try:
        model = load_model('baseline_cnn.h5')
        st.write(f"Model loaded successfully. Expected input shape: {model.input_shape}")
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return

    # Define class labels based on the dataset
    class_labels = [
        'Fresh Apple', 
        'Fresh Banana', 'Stale Banana',
        'Fresh BitterGourd', 'Stale BitterGourd',
        'Fresh Capsicum', 'Stale Capsicum',
        'Fresh Orange', 'Stale Orange',
        'Fresh Tomato', 'Stale Tomato'
    ]  # Update if your model uses different labels (e.g., ['Fresh', 'Stale'] for binary)

    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Predict and display result
        try:
            result, confidence = predict_image(uploaded_file, model, class_labels)
            st.success(f"Predicted class: {result}")
            st.write(f"Confidence: {confidence:.2%}")
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")

if __name__ == "__main__":
    main()