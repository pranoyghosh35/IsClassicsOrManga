import streamlit as st
from PIL import Image
import base64
import numpy as np
from tensorflow.keras.models import load_model

def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url(data:image/{"jpeg"};base64,{encoded_string.decode()});
            background-size: cover
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Preprocessing function
def preprocess_for_yolo(input_image, target_size=(416, 416)):
    """
    Preprocesses an input image for training YOLO.

    Parameters:
    - input_image: PIL image object or numpy array.
    - target_size: Tuple specifying the target size for the input image.

    Returns:
    - preprocessed_image: Numpy array representing the preprocessed image.
    """
    if isinstance(input_image, np.ndarray):
        input_image = Image.fromarray(input_image)

    # Resize image while maintaining aspect ratio
    width_ratio = target_size[0] / input_image.width
    height_ratio = target_size[1] / input_image.height
    resize_ratio = min(width_ratio, height_ratio)
    new_width = int(input_image.width * resize_ratio)
    new_height = int(input_image.height * resize_ratio)
    resized_image = input_image.resize((new_width, new_height))

    # Create a blank image with the target size and paste the resized image onto it
    preprocessed_image = Image.new("RGB", target_size, (128, 128, 128))
    offset = ((target_size[0] - new_width) // 2, (target_size[1] - new_height) // 2)
    preprocessed_image.paste(resized_image, offset)

    # Convert the image to a numpy array and normalize pixel values to the range [0, 1]
    preprocessed_image = np.array(preprocessed_image) / 255.0

    return preprocessed_image

def predict_image_class(image,model):
        # Preprocess image for YOLO
        preprocessed_image = preprocess_for_yolo(image)

        # Reshape image for model prediction
        preprocessed_image = np.expand_dims(preprocessed_image, axis=0)

        # Predict using the model
        y_pred_prob = model.predict(preprocessed_image)

        # Convert predicted probabilities to class labels
        y_pred_labels = np.argmax(y_pred_prob, axis=1)
        class_label = "Classics" if y_pred_labels[0] == 0 else "Manga"
        prob_predicted_class = y_pred_prob[0,y_pred_labels[0]]
        return (class_label,prob_predicted_class)
        

def main():
    st.image("ClassicsOrManga.png")
    st.title("Which Comics Style?")
    thres=0.55
    try:
        model = load_model("yolo_model_10e.keras")
    except:
        st.warning("Model temporarily offline...")
    # Upload image
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png","gif"])
    if uploaded_image is not None:
        # Display uploaded image
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_column_width=True)
	
        # Classify image
        prediction = predict_image_class(image,model)
        if prediction[1]>thres:
            st.success(f"Prediction: {prediction[0]}, Probability: {prediction[1]:.4f}")
        else:
            st.error("Not confident to classify image.")

if __name__ == "__main__":
    add_bg_from_local('bg.jpeg')  # Add background image
    main()
