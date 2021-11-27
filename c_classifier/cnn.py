import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

import numpy as np
import matplotlib.pyplot as plt

from helpers import timed


class CNN:

    def __init__(self, nfilters, sfilters):
        self.model = Sequential([
            layers.Conv2D(nfilters[0], kernel_size=(sfilters[0], sfilters[0]), padding='same', activation='relu', input_shape=(150, 150, 3)),
            layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
            layers.Conv2D(nfilters[1], kernel_size=(sfilters[1], sfilters[1]), padding='same', activation='relu'),
            layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
            layers.Conv2D(nfilters[2], kernel_size=(sfilters[2], sfilters[2]), padding='same', activation='relu'),
            layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
            layers.Flatten(),
            layers.Dropout(0.3),
            layers.Dense(512, activation='relu'),
            layers.Dense(8, activation='softmax')
        ])

    def compile(self, lr=0.0001, metrics=None):
        if metrics is None:
            metrics = ['accuracy']
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=metrics
        )

    @timed
    def train(self, X_train, y_train, X_val, y_val, batch_size=32, epochs=20, verbose=False):
        # Train Model
        history = self.model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            batch_size=batch_size,
            epochs=epochs
        )

        if verbose:
            self.model.summary()

            # Train summary
            acc = history.history['accuracy']
            val_acc = history.history['val_accuracy']

            loss = history.history['loss']
            val_loss = history.history['val_loss']

            epochs_range = range(epochs)

            plt.figure(figsize=(8, 8))
            plt.subplot(1, 2, 1)
            plt.plot(epochs_range, acc, label='Training Accuracy')
            plt.plot(epochs_range, val_acc, label='Validation Accuracy')
            plt.legend(loc='lower right')
            plt.title('Training and Validation Accuracy')

            plt.subplot(1, 2, 2)
            plt.plot(epochs_range, loss, label='Training Loss')
            plt.plot(epochs_range, val_loss, label='Validation Loss')
            plt.legend(loc='upper right')
            plt.title('Training and Validation Loss')
            plt.show()

        return history

    def predict(self, X):
        predictions = self.model.predict(X)
        return np.argmax(predictions, axis=0)


# if architecture not in ['LeNet', 'Genetic', 'GeneticV2']:
        #     raise NameError(f'{architecture} not a valid architecture')
        #
        # if architecture == 'LeNet':
        #     self.model = Sequential([
        #         layers.Conv2D(6, 5, padding='same', activation='relu', input_shape=(150, 150, 3)),
        #         layers.AveragePooling2D(),
        #         layers.Conv2D(16, 5, padding='same', activation='relu'),
        #         layers.AveragePooling2D(),
        #         layers.Flatten(),
        #         layers.Dense(120, activation='relu'),
        #         layers.Dense(84, activation='relu'),
        #         layers.Dense(8, activation='softmax')
        #     ])
        # elif architecture == 'Genetic':
        #     self.model = Sequential([
        #         layers.Conv2D(66, kernel_size=5, padding='same', activation='relu', input_shape=(150, 150, 3)),
        #         layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
        #         layers.Conv2D(40, kernel_size=5, padding='same', activation='relu'),
        #         layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
        #         layers.Conv2D(88, kernel_size=5, padding='same', activation='relu'),
        #         layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
        #         layers.Flatten(),
        #         layers.Dropout(0.3),
        #         layers.Dense(512, activation='relu'),
        #         layers.Dense(8, activation='softmax')
        #     ])
        # elif architecture == 'GeneticV2':
        #     self.model = Sequential([
        #         layers.Conv2D(74, kernel_size=9, padding='same', activation='relu', input_shape=(150, 150, 3)),
        #         layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
        #         layers.Conv2D(27, kernel_size=3, padding='same', activation='relu'),
        #         layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
        #         layers.Conv2D(23, kernel_size=2, padding='same', activation='relu'),
        #         layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
        #         layers.Flatten(),
        #         layers.Dropout(0.3),
        #         layers.Dense(512, activation='relu'),
        #         layers.Dense(8, activation='softmax')
        #     ])
