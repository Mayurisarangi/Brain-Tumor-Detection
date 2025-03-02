Brain Tumor Detection using CNN

Project Overview

This project implements a Convolutional Neural Network (CNN) model to detect brain tumors from MRI scans. It utilizes TensorFlow and Keras for deep learning, along with data preprocessing and visualization techniques to enhance model performance. The project also includes a Streamlit-based web application for real-time tumor prediction.

Features
•	Download and preprocess brain tumor MRI dataset from Kaggle.
•	Organize images into "tumor" and "no tumor" classes.
•	Data augmentation and preprocessing.
•	CNN-based deep learning model for classification.
•	Model training, validation, and evaluation.
•	Streamlit web app for real-time predictions.

Dataset
The dataset is sourced from Kaggle: Brain Tumor MRI Dataset.

Installation

Prerequisites
Ensure you have the following installed:
•	Python 3.x
•	TensorFlow
•	Keras
•	NumPy
•	Matplotlib
•	Streamlit
•	PIL

Setting up the Environment

!pip install kaggle numpy matplotlib tensorflow keras streamlit

Downloading the Dataset
from google.colab import files
files.upload()
! mkdir ~/.kaggle
! cp kaggle.json ~/.kaggle/
! chmod 600 ~/.kaggle/kaggle.json
! kaggle datasets download -d masoudnickparvar/brain-tumor-mri-dataset

Data Preprocessing
•	Extract dataset
•	Organize into tumor and no_tumor folders
•	Convert images to grayscale
•	Train-test split (70%-30%)

Model Architecture
The CNN model consists of:
•	Three convolutional layers with ReLU activation
•	MaxPooling layers to reduce spatial dimensions
•	Fully connected dense layers for classification
•	Softmax activation for output


Training the Model
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

Model Evaluation
test_score = model.evaluate(X_test, y_test)
print("Test Loss:", test_score[0])
print("Test Accuracy:", test_score[1])

Running the Web Application

Install Streamlit

!pip install streamlit

Run Streamlit App

!streamlit run app.py & npx localtunnel --port 8501

Usage
1.	Upload an MRI scan (JPG, PNG) in the Streamlit UI.
2.	The model processes the image and predicts whether a tumor is present.
3.	The prediction result is displayed.

   
Model Deployment
The trained model (Bestmodel.h5) is saved for deployment and can be loaded in the web app for real-time predictions.


Contributing
Feel free to contribute by improving the model, optimizing performance, or enhancing the web app UI.

License
This project is open-source and available under the MIT License.
