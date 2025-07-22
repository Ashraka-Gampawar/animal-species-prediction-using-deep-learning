# animal-species-prediction-using-deep-learning
This project uses deep learning (CNN) models—ResNet152, ResNet50, and VGG16—to classify animal species from images. It features a Streamlit web app for real-time predictions, model selection, and result logging.

🐾 Animal Species Prediction using CNN (ResNet152, ResNet50, VGG16)
This project is a deep learning-based web application for automatic animal species classification using Convolutional Neural Networks (CNN). It allows users to upload an animal image and receive real-time predictions along with confidence scores.

The system is built using Streamlit for the frontend and leverages pre-trained CNN models—ResNet152, ResNet50, and VGG16—from Keras applications with ImageNet weights for feature extraction and prediction. Among these, ResNet152 is used as the primary model due to its deep architecture and superior classification performance.

🔍 Key Features
Upload and classify animal species in real-time

Supports ResNet152, ResNet50, and VGG16 models

Image preprocessing, top-1 prediction decoding, and confidence scoring

CSV logging of each prediction with timestamp and model used

Simple and interactive Streamlit UI

Model selection dropdown and result download option

📌 Use Cases
Wildlife conservation and monitoring

Biodiversity research

Educational tools and citizen science platforms

🚀 Tech Stack
Python

TensorFlow / Keras

Streamlit

OpenCV, NumPy, Pandas
