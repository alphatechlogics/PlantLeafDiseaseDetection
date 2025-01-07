# 🌿 Leaf Disease Detection

## 🚀 Project Overview

This project is a Deep Learning-based Leaf Disease Detection System that utilizes transfer learning to identify and classify diseases in plant leaves. The app leverages a pre-trained model to provide high-accuracy predictions for 33 different types of leaf diseases.

## 📋 Features

- **Image Upload**: Upload a clear image of a leaf.
- **Disease Detection**: Predicts the disease with confidence scores.
- **Interactive UI**: Built with Streamlit for an engaging user experience.

## 💂️ File Structure

- **main.py**: Contains the code for the Streamlit application.
- **Training/model/Leaf Deases(96,88).h5**: Pre-trained Keras model used for predictions.
- **Leaf_Disease_Train_Notebook.ipynb**: Jupyter notebook detailing the training process and model evaluation.

## 📊 Dataset Details

### 🌱 Dataset Source

The dataset used for this project is the New Plant Diseases Dataset.

### 📈 Dataset Overview

- **Classes**: 38 (including healthy and diseased categories for various crops).
- **Number of Images**: Over 87,000 augmented images.
- **Augmentation**: Images were resized, rotated, flipped, and cropped to enhance model generalization.
- **Crops Covered**: Includes apples, cherries, corn, grapes, peaches, peppers, potatoes, strawberries, and tomatoes.

## 🧠 Model Details

### 📋 Architecture

The model employs Transfer Learning using a pre-trained Convolutional Neural Network (CNN). Specifics include:

- **Base Model**: EfficientNetB0.
- **Input Size**: 150x150 RGB images.
- **Layers Added**:
  - Fully connected layers
  - Dropout layers for regularization
  - Softmax activation for multiclass classification

### 🏃️ Training

- **Loss Function**: Categorical Crossentropy
- **Optimizer**: Adam with learning rate scheduling
- **Metrics**: Accuracy
- **Batch Size**: 32
- **Epochs**: 25

## 📊 Performance Metrics

- **Accuracy**: Achieved over 95% accuracy on the validation set.
- **Loss**: Minimal training and validation loss with no signs of overfitting.

## 📝 Jupyter Notebook Highlights

The training process is documented in the `Leaf_Disease_Train_Notebook.ipynb` file, which includes:

- **Data Preprocessing**: Loading, augmentation, and splitting the dataset
- **Model Building**: Implementing and fine-tuning the transfer learning model
- **Training**: Visualization of loss and accuracy
- **Evaluation**: Confusion matrix and class-wise performance

## 📦 Prerequisites

- Install **Python 3.10** or higher.

## 💾 Installation

Follow these steps to set up the project locally:

1. **Install `virtualenv`** (if not already installed):

   ```bash
   pip install virtualenv
   ```

2. **Create a Virtual Environment**:

   Create a new virtual environment. You can name it as you prefer; here, we'll use `leafdisease-venv`:

   ```bash
   virtualenv leafdisease-venv
   ```

3. **Activate the Virtual Environment**:

   - **Windows**:

     ```bash
     leafdisease-venv\Scripts\activate
     ```

   - **macOS/Linux**:

     ```bash
     source leafdisease-venv/bin/activate
     ```

4. **Clone the Repository**:

   Clone the repository into the directory where you created the virtual environment:

   ```bash
   git clone https://github.com/alphatechlogics/PlantLeafDiseaseDetection.git
   ```

5. **Navigate to the Project Directory**:

   ```bash
   cd PlantLeafDiseaseDetection
   ```

6. **Install Project Requirements**:

   Install all necessary dependencies using `pip`:

   ```bash
   pip install -r requirements.txt
   ```

7. Run the Streamlit app:

   ```bash
   streamlit run main.py
   ```

8. Open the app in your browser (default: http://localhost:8501).
