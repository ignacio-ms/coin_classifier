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
            layers.Conv2D(nfilters[0], kernel_size=(sfilters[0], sfilters[0]), padding='same', input_shape=(150, 150, 3)),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.MaxPooling2D(pool_size=(2, 2)),

            layers.Conv2D(nfilters[1], kernel_size=(sfilters[1], sfilters[1]), padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.MaxPooling2D(pool_size=(2, 2)),

            layers.ZeroPadding2D((1, 1)),
            layers.Conv2D(nfilters[2], kernel_size=(sfilters[2], sfilters[2]), padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.MaxPooling2D(pool_size=(2, 2)),

            layers.ZeroPadding2D((1, 1)),
            layers.Conv2D(nfilters[3], kernel_size=(sfilters[3], sfilters[3]), padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu'),

            layers.ZeroPadding2D((1, 1)),
            layers.Conv2D(nfilters[4], kernel_size=(sfilters[4], sfilters[4]), padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.MaxPooling2D(pool_size=(2, 2)),

            layers.Flatten(),
            layers.Dense(1024),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.Dropout(0.5),

            layers.Dense(2048),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.Dropout(0.5),

            layers.Dense(8),
            layers.BatchNormalization(),
            layers.Activation('softmax'),
        ])

    def compile(self, lr=1e-3, metrics=None):
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
            ReduceLROnPlateau(monitor="val_accuracy", patience=2, factor=0.1, verbose=1, min_lr=1e-6),
            EarlyStopping(monitor="val_accuracy", patience=5, verbose=1)
        ]
        if save:
            callbacks.append(ModelCheckpoint(filepath='D:\\model_alex.h5', monitor="val_accuracy", verbose=1, save_best_only=True))

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

    def load(self, path):
        self.model.load_weights(path)

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
