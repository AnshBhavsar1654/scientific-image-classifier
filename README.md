# scientific-image-classifier

Scientific Image Classification using Deep Learning
This repository contains a deep learning model implemented in Python using TensorFlow and Keras to classify biological images into four different types: Microscopy, BlotGel, FACS, and Macroscopy. The model is trained on a dataset named Biofors2, and it utilizes the DenseNet121 architecture pre-trained on ImageNet for feature extraction.

Dataset
The dataset consists of images categorized into four subdirectories, namely Microscopy, BlotGel, FACS, and Macroscopy. To ensure a balanced dataset, Synthetic Minority Over-sampling Technique (SMOTE) is applied to handle class imbalances.

Model Architecture
The deep learning model architecture consists of the following layers:
Pre-trained DenseNet121 base model with weights trained on ImageNet
Global Average Pooling layer
Fully connected dense layers with ReLU activation
Dropout layer to prevent overfitting
Output layer with softmax activation for multi-class classification

Training
The data is split into training and test sets with a ratio of 80:20.
Data augmentation techniques such as rotation, shifting, shearing, zooming, and horizontal flipping are applied to enhance model generalization.
The model is trained using the Adam optimizer with a learning rate of 0.0001 and sparse categorical cross-entropy loss.
Model training is performed for 10 epochs with a batch size of 32.

Evaluation
Training and validation accuracy and loss are plotted to visualize the model's performance during training.
Model evaluation is performed on the test set, and metrics including test loss, test accuracy, and classification report are printed.

Inference
The trained model is utilized to predict the types of biological images provided in a test directory. The predicted types are displayed along with the respective images.
