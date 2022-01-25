# Build a simple DNN with TensorFlow and Keras
# Reference: https://github.com/tensorflow/docs/blob/master/site/en/tutorials/keras/classification.ipynb
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

print(tf.__version__)

#Import the dataset - Fashion MNIST
fashion_mnist = tf.keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

#Data Preprocessing 
plt.figure()
plt.colorbar()
plt.grid(False)

train_images = train_images / 255.0

test_images = test_images / 255.0

plt.figure(figsize=(10,10))

#Build the model:
## Step 1: Set up the layers
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
])

## Step 2: Compile the model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Train the model
model.fit(train_images, train_labels, epochs=10)

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

print('\nTest accuracy:', test_acc)