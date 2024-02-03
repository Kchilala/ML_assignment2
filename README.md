# Pose Classification Project
## Project Overview
This project focuses on classifying different poses based on the keypoints extracted from the human body using machine learning techniques. The dataset comprises 33 keypoints of the body for 17 unique pose classes. We utilized Support Vector Machines (SVM) for the classification task, aiming to accurately predict the pose based on the given keypoints.

## Getting Started
### Prerequisites
Python 3.x
NumPy
pandas
scikit-learn
Matplotlib (for data visualization)
### Installation
Clone the repository to your local machine:
git clone https://your-repository-link
Install the required Python packages:
pip install numpy pandas scikit-learn matplotlib
### Running the Code
Navigate to the project directory:

bash

cd path/to/your/cloned/repo
Run the main script to train the model and evaluate its performance:


python pose_classification.py
## Data Preparation
The dataset is organized into folders, each representing a different pose class. Each .npy file within these folders contains an array of keypoints for a single pose instance. We preprocessed this data by flattening the keypoints into feature vectors and normalized the features for better model performance.

## Model Training and Evaluation
We trained 17 binary SVM models, one for each pose class, using a one-vs-rest strategy. The models were evaluated using accuracy, precision, recall, and F1-score metrics on both validation and test datasets.

## Deployment
The trained models are saved using joblib and can be deployed for predicting new pose instances. A simple Flask or Django app can serve as an API endpoint for pose classification.

### How to Deploy the Model
## Load the trained model:


import joblib
model = joblib.load('path/to/saved_model.pkl')
Predict new instances:

new_pose_keypoints = [...]  # Your new pose keypoints here
prediction = model.predict([new_pose_keypoints])
print("Predicted Pose Class:", prediction)
Discussion
Methodology
I chose SVM due to its effectiveness in high-dimensional spaces and its ability to handle non-linearly separable data using kernel tricks. The challenge was to manage the multi-class classification, which we addressed using binary classifiers for each class.

# Limitations
The dataset is imbalanced, which might bias the model towards classes with more samples.
Only keypoints are used as features; incorporating additional features like distances and angles between keypoints might improve the model.
Future Work
Explore deep learning models, particularly CNNs, for pose classification.
Implement data augmentation techniques to address the class imbalance issue.
Deploy the model as a web or mobile application for real-time pose classification.
Contributors:
Keci Chilala


