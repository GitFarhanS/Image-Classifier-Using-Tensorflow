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

# Load the saved model for testing
loaded_model = keras.models.load_model('trained_model.h5')

# Evaluate the loaded model on the test data
testLoss, testAcc = loaded_model.evaluate(testImages,  testLabels, verbose=2)

print('\nTest accuracy:', testAcc)

# Use the loaded model for making predictions
probabilityModel = tf.keras.Sequential([loaded_model, tf.keras.layers.Softmax()])

prediction = probabilityModel.predict(testImages)

# Function to plot an image, its predicted label, and true label
def plot_image(i, predictionsArray, trueLabel, img):
  trueLabel, img = trueLabel[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  plt.imshow(img, cmap=plt.cm.binary)

  predictedLabel = np.argmax(predictionsArray)
  if predictedLabel == trueLabel:
    color = 'green'
  else:
    color = 'red'

  plt.xlabel("{} {:2.0f}% ({})".format(classNames[predictedLabel],
                                100*np.max(predictionsArray),
                                classNames[trueLabel]),
                                color=color)

# Function to plot the prediction probabilities as a bar chart
def plot_value_array(i, predictionsArray, trueLabel):
  trueLabel = trueLabel[i]
  plt.grid(False)
  plt.xticks(range(10))
  plt.yticks([])
  thisplot = plt.bar(range(10), predictionsArray, color="#777777")
  plt.ylim([0, 1])
  predictedLabel = np.argmax(predictionsArray)

  thisplot[predictedLabel].set_color('red')
  thisplot[trueLabel].set_color('green')

# Plot the first X test images, their predicted labels, and the true labels.
# Colour correct predictions in blue and incorrect predictions in red.
num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i, prediction[i], testLabels, testImages)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i, prediction[i], testLabels)
plt.tight_layout()
plt.show()
