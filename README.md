# Perceptron Mini-Project

This project implements a simple **Perceptron** for binary classification using the Iris dataset, specifically classifying **Iris-versicolor** and **Iris-virginica**. The perceptron is trained on a subset of the Iris dataset and evaluated on a separate test set.

## 📂 Dataset

- **Training set**: `perceptron.data`  
- **Test set**: `perceptron.test.data`  
- Both files are in CSV format. The last column in each row represents the class label.

## Features

- **Dynamic dataset loading**: Accepts any CSV file where the last column is the class.
- **Customizable learning rate**: Choose your preferred learning rate at runtime.
- **Randomized initialization**: Weights and bias are randomly initialized in the range [0, 1].
- **Flexible training loop**: Supports configurable iteration count for learning.
- **Accuracy evaluation**: Automatically computes accuracy on the test set after training.
- **Simple UI**: Manually input vectors to classify using the trained model.

## Notes
- A small learning rate (e.g. `0.01`) combined with a higher number of iterations improves results.
- Iris-versicolor and Iris-virginica are **not linearly separable**, so the training error may never reach zero — but good accuracy is still achievable.
