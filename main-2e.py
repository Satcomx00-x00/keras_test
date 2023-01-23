#!/usr/bin/env python3
# *-* coding: utf-8 *-*
from keras.utils import plot_model

import tensorflow as tf
import numpy as np
from tensorflow import keras
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

(x_train, y_train), (x_test, y_test) = tfds.as_numpy(tfds.load('emnist/letters',
                                                               data_dir="C:/",
                                                               split=[
                                                                   'train', 'test'],
                                                               as_supervised=True,
                                                               batch_size=-1))

# Transformation de matrices 28x28 en vecteurs
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

# Normaliser les données
x_train = x_train / 255.0
x_test = x_test / 255.0

# Rotating the image by 90 degrees counterclockwise and then flipping it horizontally.
x_train = np.fliplr(np.rot90(x_train, k=-1, axes=(1, 2)
                             ).reshape(x_train.shape[0], 28, 28, 1))
x_test = np.fliplr(np.rot90(x_test, k=-1, axes=(1, 2)
                            ).reshape(x_test.shape[0], 28, 28, 1))

# To counter => IndexError: index 25 is out of bounds for axis 1 with size 10
# Converting the labels of the digits to the labels of the letters.
y_train = np.where(y_train >= 10, 0, y_train)
y_test = np.where(y_test >= 10, 0, y_test)

# to_categorical => convert a class vector (integers) to binary class matrix.
# Converting the labels of the digits to the labels of the letters.
y_train = tf.keras.utils.to_categorical(y_train, 26)
y_test = tf.keras.utils.to_categorical(y_test, 26)


print(x_train.shape, y_train.shape)
# Définir le modèle
model = keras.Sequential()
model.add(keras.layers.Dropout(0.25))
model.add(keras.layers.Conv2D(32, kernel_size=(3, 3),
          activation='relu', input_shape=(28, 28, 1)))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(64, activation='relu'))
model.add(keras.layers.Dense(26, activation='softmax'))

# # Compiler le modèle
model.compile(optimizer='adam',
              loss='categorical_crossentropy', metrics=['accuracy'])
# ValueError: Input 0 of layer "sequential" is incompatible with the layer: expected shape=(None, 28, 28, 1), found shape=(32, 784)
model.fit(x_train, y_train, epochs=20, batch_size=32)
plot_model(model, to_file='model_2.png', show_shapes=True, show_layer_names=True)

# Évaluer le modèle sur les données de test
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print('Test accuracy:', test_acc)
print(f"Test loss: {test_loss:.2f}")
print(f"Test accuracy: {test_acc:.2f}")
print(f"Test error: {100 - test_acc * 100:.2f}%")

# Test accuracy: 0.9132432341575623
# Test loss: 0.27
# Test accuracy: 0.91
# Test error: 8.68%