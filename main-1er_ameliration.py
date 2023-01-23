#!/usr/bin/env python3
# *-* coding: utf-8 *-*
from keras.utils import plot_model

# Importation des bibliothèques nécessaires
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
# Charger la base de données EMNIST et la diviser en ense
import tensorflow_datasets as tfds
from tensorflow.keras.layers import Dense
# ImportError: cannot import name 'emnist' from 'tensorflow.keras.datasets'
# Solution: pip install --upgrade tensorflow
from tensorflow.keras.models import Sequential


# mbles d'entraînement et de test
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


# use np.fliplr(np.rot90(x_train[i], k=-1, axes=(0, 1)).reshape(28, 28)) for all images in x_train
# Faire tourner les images de 90 degrés dans le sens antihoraire.
x_train = np.fliplr(np.rot90(x_train, k=-1, axes=(1, 2)
                             ).reshape(x_train.shape[0], 28, 28, 1))
x_test = np.fliplr(np.rot90(x_test, k=-1, axes=(1, 2)
                            ).reshape(x_test.shape[0], 28, 28, 1))

# Transformation de matrices 28x28 en vecteurs
x_train = x_train.reshape(x_train.shape[0], 28*28)
x_test = x_test.reshape(x_test.shape[0], 28*28)

# To counter => IndexError: index 25 is out of bounds for axis 1 with size 10
# Conversion du vecteur de classe (entiers) en matrice de classe binaire.
y_train = np.where(y_train >= 26, 0, y_train)
y_test = np.where(y_test >= 26, 0, y_test)

# to_categorical => convert a class vector (integers) to binary class matrix.
# Conversion du vecteur de classe (entiers) en matrice de classe binaire.
y_train = tf.keras.utils.to_categorical(y_train, 26)
y_test = tf.keras.utils.to_categorical(y_test, 26)


# Définir le modèle du perceptron
model = Sequential()
# Ajout d'une couche entièrement connectée avec 50 neurones au modèle. La dimension d'entrée est
# 28*28, qui est la taille de l'image aplatie. La fonction d'activation est ReLU.
from keras import layers
from keras import models
model.add(tf.keras.layers.Dense(
    units=512, input_shape=(784,), activation='relu'))
# model.add(tf.keras.layers.Dropout(0.25))
model.add(tf.keras.layers.Dense(units=256, activation='relu'))
model.add(tf.keras.layers.Dense(units=128, activation='relu'))
model.add(tf.keras.layers.Dense(units=64, activation='relu'))
model.add(tf.keras.layers.Dense(units=32, activation='relu'))
model.add(tf.keras.layers.Dense(26, activation='softmax'))

# model.add(tf.keras.layers.Dense(units=26, activation='softmax'))
# Utilisation de la régularisation Dropout
# Test loss: 2.1488821133971214
# Test accuracy: 90.05405306816101
# Test error: 9.94594693183899
# Utilisation de l'optimiseur Adamax
model.compile(optimizer='adam',
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

# Entraînement sur des données supplémentaires
history = model.fit(x_train, y_train, epochs=10, batch_size=64)
# sigmoid
# Test loss: 36.34302616119385
# Test accuracy: 89.61486220359802
# Test error: 10.385137796401978

# Entraînement du modèle sur les données d'entraînement pour 10 époques avec une taille de lot de 128.
# model.fit(x_train, y_train, epochs=10, batch_size=32)

plot_model(model, to_file='model1-2.png', show_shapes=True, show_layer_names=True)

# Évaluation du modèle sur les données de test.
loss, accuracy = model.evaluate(x_test, y_test)

# get in percent the accuracy and the error rate and the loss
print("# Test loss:", 100*loss)
print("# Test accuracy:", 100*accuracy)
print("# Test error:", 100*(1-accuracy))


# Test loss: 0.3723438084125519
# Test accuracy: 0.8762162327766418
# Test error: 0.12378376722335815
