import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

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
            layers.Dropout(0.3),
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.Dense(8, activation='softmax')
        ])

    def compile(self, lr=1e-4, metrics=None):
        if metrics is None:
            metrics = ['accuracy']
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=metrics
        )

    @timed
    def train(self, X_train, y_train, X_val, y_val, batch_size=32, epochs=20, save=False, verbose=False):
        callbacks = [
            ReduceLROnPlateau(monitor="val_loss", patience=3, factor=0.1, verbose=1, min_lr=1e-6),
            EarlyStopping(monitor="val_loss", patience=5, verbose=1)
        ]
        if save:
            callbacks = [
                ModelCheckpoint(filepath='models/model.h5', verbose=1, save_best_only=True),
                ReduceLROnPlateau(monitor="val_loss", patience=3, factor=0.1, verbose=1, min_lr=1e-6),
                EarlyStopping(monitor="val_loss", patience=5, verbose=1)
            ]

        # Train Model
        history = self.model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks
        )

        if verbose:
            self.model.summary()

            # Train summary
            acc = history.history['accuracy']
            val_acc = history.history['val_accuracy']

            loss = history.history['loss']
            val_loss = history.history['val_loss']

            plt.figure(figsize=(8, 8))
            plt.subplot(1, 2, 1)
            plt.plot(acc, label='Training Accuracy')
            plt.plot(val_acc, label='Validation Accuracy')
            plt.legend(loc='lower right')
            plt.title('Training and Validation Accuracy')

            plt.subplot(1, 2, 2)
            plt.plot(loss, label='Training Loss')
            plt.plot(val_loss, label='Validation Loss')
            plt.legend(loc='upper right')
            plt.title('Training and Validation Loss')
            plt.show()

        return history

    def predict_per_class(self, X, y, verbose=False):
        pred = self.model.predict(X)
        pred = np.argmax(pred, axis=1)

        prob_per_class = []
        for c in np.unique(y):
            c_pred = np.sum(np.where(pred[y == c] == y[y == c], 1, 0))
            prob_per_class.append(c_pred / np.sum(np.where(y == c, 1, 0)))

        if verbose:
            plt.bar(np.unique(y), prob_per_class)
            plt.title('Accuracy predictions per class')
            plt.xlabel('Classes')
            plt.ylabel('Accuracy')
            plt.show()

        return prob_per_class

    def predict(self, X):
        predictions = self.model.predict(X)
        return np.argmax(predictions, axis=1)
