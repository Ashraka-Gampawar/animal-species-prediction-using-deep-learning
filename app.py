import streamlit as st
from tensorflow.keras.applications import (
    ResNet152,
    ResNet50,
    VGG16,
    resnet,
    vgg16,
)
from tensorflow.keras.preprocessing import image
import numpy as np
import csv
import os
from datetime import datetime
import pandas as pd

# Load all models once and cache
@st.cache_resource
def load_all_models():
    return {
        "ResNet152": ResNet152(weights="imagenet"),
        "ResNet50": ResNet50(weights="imagenet"),
        "VGG16": VGG16(weights="imagenet"),
    }

# Unified preprocessing and decoding logic
def get_preprocessor_and_decoder(model_name):
    if model_name in ["ResNet152", "ResNet50"]:
        return resnet.preprocess_input, resnet.decode_predictions
    elif model_name == "VGG16":
        return vgg16.preprocess_input, vgg16.decode_predictions

# Predict species
def predict_species(img, model_name, models_dict):
    model = models_dict[model_name]
    preprocess_input, decode_predictions = get_preprocessor_and_decoder(model_name)

    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    preds = model.predict(x, verbose=0)  # suppress verbose
    decoded_preds = decode_predictions(preds, top=1)[0]
    species_name = decoded_preds[0][1].replace("_", " ")
    accuracy = decoded_preds[0][2]
    return species_name, accuracy

# Save results
def save_results_to_csv(results):
    file_name = "results.csv"
    with open(file_name, "a+", newline="") as f:
        writer = csv.writer(f)
        for result in results:
            writer.writerow([result[0], result[1], result[2], result[3], result[4]])
    return file_name

# Display previous results
def display_previous_results():
    try:
        df = pd.read_csv("results.csv", header=None, names=["Date", "Time", "Species", "Accuracy", "Model"])
        st.write("Previous Results:")
        st.dataframe(df)
    except FileNotFoundError:
        st.warning("No previous results found.")

# Main app
def main():
    st.title("🐾 Animal Species Prediction Using Deep Learning")

    st.sidebar.title("Options")
    if st.sidebar.button("Display Previous Results"):
        display_previous_results()

    selected_model = st.sidebar.selectbox("Select Model", ["ResNet152", "ResNet50", "VGG16"])

    uploaded_file = st.file_uploader("📤 Upload an image...", type=["jpg", "jpeg", "png"])

    # Load all models once
    models_dict = load_all_models()

    if uploaded_file is not None:
        img = image.load_img(uploaded_file, target_size=(224, 224))
        st.image(img, caption="Uploaded Image.", use_column_width=True)

        if st.button("🔍 Predict"):
            species_name, accuracy = predict_species(img, selected_model, models_dict)
            st.success(f"Species: **{species_name}**")
            st.info(f"Accuracy: **{accuracy * 100:.2f}%**")

            now = datetime.now()
            results = [(now.strftime("%Y-%m-%d"), now.strftime("%H:%M:%S"), species_name, accuracy, selected_model)]
            file_name = save_results_to_csv(results)
            st.success(f"✅ Results saved to `{file_name}`")

            st.session_state.results = file_name

    if "results" in st.session_state:
        with open(st.session_state.results, "rb") as f:
            data = f.read()
        st.sidebar.download_button(
            "⬇️ Download Results CSV",
            data,
            file_name=os.path.basename(st.session_state.results),
            key="download_button",
        )

if __name__ == "__main__":
    main()


