# FashionVision-CNN 👗👟🧥

A Convolutional Neural Network (CNN) model built using TensorFlow and Keras to classify images from the Fashion-MNIST dataset. The dataset consists of 10 different classes of clothing and footwear items.

---

## 📌 Features

- Load and visualize the Fashion-MNIST dataset
- Build a multi-layer CNN architecture
- Use dropout to prevent overfitting
- Evaluate the model using accuracy, confusion matrix, and classification report
- Visualize training history and prediction results

---

## 🧠 Model Architecture

- Conv2D + MaxPooling + Dropout (3 blocks)
- Flatten layer
- Dense layer with ReLU activation
- Final Dense layer with softmax (for 10 classes)

---

## 📊 Evaluation Metrics

- Accuracy: ~92-93%
- Visual Confusion Matrix
- Classification Report with precision, recall, and F1-score

---

## 🧪 Requirements

- Python 3.x
- TensorFlow
- NumPy
- Matplotlib
- Seaborn
- scikit-learn

Install dependencies using:
```bash
pip install tensorflow numpy matplotlib seaborn scikit-learn
