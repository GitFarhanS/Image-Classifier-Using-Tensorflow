# Importing required libraries
import tensorflow as tf
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt

# Loading the Fashion MNIST dataset
fashionMnist = keras.datasets.fashion_mnist

# Splitting the dataset into train and test sets
((trainImages, trainLabels), (testImages, testLabels)) = fashionMnist.load_data()

# Defining class names for visualisation
classNames = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Preprocessing the images by normalising pixel values
trainImages = trainImages / 255.0
testImages = testImages / 255.0

# Visualising the training images
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(trainImages[i], cmap=plt.cm.binary)
    plt.xlabel(classNames[trainLabels[i]])
plt.show()

# Defining the neural network model
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),   # Flatten the input images
    tf.keras.layers.Dense(128, activation='relu'),    # Hidden layer with 128 units and ReLU activation
    tf.keras.layers.Dense(10)                         # Output layer with 10 units (for 10 classes)
])

# Compiling the model with optimiser, loss function, and evaluation metric
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Training the model
model.fit(trainImages, trainLabels, epochs=10)

# Saving the trained model to a file
model.save('trained_model.h5')
