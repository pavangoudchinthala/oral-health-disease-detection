import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
from tensorflow.keras.preprocessing import image

# Load the trained model
MODEL_PATH = r"C:\Users\pavan\OneDrive\Desktop\3(2)_ MINI_PROJECT\Oral_Health_Early_Disease_Detection_1\Oral_Health_Early_Disease_Detection_1\oral_disease_prediction_cnn.h5"
model = load_model(MODEL_PATH)

# Define class labels
CLASS_LABELS = ["Mouth Ulcer", "Tooth Discoloration"]

# Prediction function
def predict_oral_health_disease(uploaded_file, model):
    """Preprocess and predict the oral health disease from the uploaded image."""
    try:
        img = Image.open(uploaded_file).convert("RGB")
        img = img.resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Display loading animation
        with st.spinner("🔍 Analyzing image... Please wait!"):
            prediction = model.predict(img_array)

        # Determine class
        if prediction.shape[-1] > 1:
            predicted_class = np.argmax(prediction)
        else:
            predicted_class = int(prediction[0] > 0.5)

        class_label = CLASS_LABELS[predicted_class]
        confidence = float(prediction[0][predicted_class]) * 100 if prediction.shape[-1] > 1 else float(prediction[0]) * 100

        return class_label, confidence, img

    except Exception as e:
        return f"Error: {e}", None, None

# Streamlit UI Setup
st.set_page_config(page_title="Oral Health Diagnosis", page_icon="🦷", layout="wide")

# Custom CSS for background and styling
st.markdown(
    """
    <style>
        .stApp {
            background: url("https://img.freepik.com/premium-photo/abstract-blue-medical-background-healthcare-science-concept_34629-753.jpg");
            background-size: cover;
        }
        .title {
            font-size: 42px;
            font-weight: bold;
            text-align: center;
            color: white;
            padding: 20px;
            text-shadow: 2px 2px 8px rgba(0,0,0,0.5);
        }
        .upload-box {
            background: rgba(255, 255, 255, 0.9);
            padding: 20px;
            border-radius: 10px;
        }
        .stButton>button {
            background-color: #007bff !important;
            color: white !important;
            border-radius: 8px;
            padding: 10px;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Page Title
st.markdown('<h1 class="title">🦷 Oral Disease Detection</h1>', unsafe_allow_html=True)
st.markdown("### 📷 Upload an image to detect oral health conditions.")

# Sidebar with instructions
st.sidebar.header("📝 Instructions")
st.sidebar.markdown(
    """
    - Upload a **clear image** of the affected area.
    - Ensure proper lighting for **better accuracy**.
    - Supported formats: **JPG, PNG, JPEG**.
    """
)

# File Upload Box with Background
st.markdown('<div class="upload-box">', unsafe_allow_html=True)
uploaded_image = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])
st.markdown('</div>', unsafe_allow_html=True)

if uploaded_image is not None:
    class_label, confidence, img = predict_oral_health_disease(uploaded_image, model)

    if img:
        col1, col2 = st.columns(2)

        with col1:
            st.image(img, caption="📸 Uploaded Image", width=400, use_column_width=True)

        with col2:
            st.success(f"✅ **Prediction:** {class_label}")
            st.write(f"**Confidence Level:** {confidence:.2f}%")

            # Disease-specific recommendations
            if class_label == "Tooth Discoloration":
                st.warning("🟡 **Tooth Discoloration**: Caused by smoking, coffee, or aging. Try whitening toothpaste or consult a dentist for professional cleaning.")
                
                # Specific Preventive Tips for Tooth Discoloration
                st.markdown(
                    """
                    ### **🦷 Prevention Tips for Tooth Discoloration:**
                    - ✅ **Use whitening toothpaste** containing fluoride.
                    - 🚫 **Avoid excessive coffee, tea, and smoking**.
                    - 🥕 **Increase intake of crunchy vegetables** (like carrots) to help clean teeth.
                    - 🦷 **Consider professional dental cleaning** every 6 months.
                    """
                )

            else:
                st.warning("🔴 **Mouth Ulcer**: Could be due to stress, infections, or vitamin deficiency. Avoid spicy foods and use an antiseptic mouthwash.")
                
                # Specific Preventive Tips for Mouth Ulcer
                st.markdown(
                    """
                    ### **💊 Prevention Tips for Mouth Ulcers:**
                    - 🥦 **Eat a balanced diet** rich in vitamins (especially B12 & iron).
                    - 🚫 **Avoid spicy, acidic, and very hot foods**.
                    - 🥛 **Use honey or coconut oil** as a natural remedy.
                    - 🛑 **Reduce stress** through meditation or relaxation techniques.
                    - 🚰 **Drink plenty of water** to keep your mouth hydrated.
                    """
                )

    else:
        st.error("⚠️ Could not process the image. Please upload a valid image.")
