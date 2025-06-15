# 😷 Face Mask Detector using CNN

This project is a simple yet effective deep learning model built using **TensorFlow Keras** in a **Google Colab** notebook. It detects whether a person is wearing a face mask or not from an image, leveraging convolutional neural networks (CNN) for binary image classification.

---

## 📁 Dataset

The dataset used is the **Face Mask Dataset** from **Kaggle**, which includes labeled images of individuals with and without face masks.

- 📸 Image Size: Resized to `128x128x3`
- 🧾 Labels: `0` for **No Mask**, `1` for **Mask**

---

## 🧠 Model Architecture

The model is a **Sequential CNN**, consisting of the following layers:

```python
model = keras.Sequential()

# Conv Layer 1
model.add(keras.layers.Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=(128,128,3)))
model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))

# Conv Layer 2
model.add(keras.layers.Conv2D(64, kernel_size=(3,3), activation='relu'))
model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))

# Flatten and Dense Layers
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(128, activation='relu'))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(64, activation='relu'))
model.add(keras.layers.Dropout(0.5))

# Output Layer
model.add(keras.layers.Dense(2, activation='sigmoid'))


#Compilation

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['acc'])


#🚀 Features
Simple and efficient CNN model

Real-time mask detection possible with further integration

Built entirely in Google Colab

Uses Dropout to reduce overfitting



#💡 How It Works
Images are preprocessed and resized to 128x128 pixels.

The CNN model is trained on labeled data (Mask / No Mask).

The trained model outputs a prediction: Mask 😷 or No Mask


🧪 Future Improvements
Real-time webcam mask detection using OpenCV

Model optimization using transfer learning (e.g., MobileNet, ResNet)

Deployment using Streamlit or Flask


---Thank You---
