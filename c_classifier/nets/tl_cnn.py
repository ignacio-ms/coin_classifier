from tensorflow.keras.applications.vgg16 import VGG16
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from focal_loss import SparseCategoricalFocalLoss
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    BatchNormalization, Dropout,
    Flatten, Dense
)

import numpy as np
import matplotlib.pyplot as plt


class TransferVGG:

    def __init__(self):
        self.input_shape = (150, 150, 3)

        self.model = None
        self.base_model = VGG16(include_top=False, weights='imagenet', input_shape=self.input_shape)
        for i in range(4):
            self.base_model.layers.pop()

    def build_top(self):
        """
        This function loads the top of the VGG16 pretrained model.
        """
        x = self.base_model.output
        x = Flatten()(x)
        x = Dense(4056, activation='elu', kernel_initializer='he_uniform')(x)
        x = BatchNormalization()(x)
        x = Dropout(rate=0.5)(x)
        predictions = Dense(8, activation='softmax')(x)

        self.model = Model(inputs=self.base_model.inputs, outputs=predictions)

        for layer in self.model.layers[:-8]:
            layer.trainable = False

    def compile(self, lr=1e-3, metrics=None):
        """
        This function compiles the tensorflow model
        :param lr: Learning Rate
        :param metrics: Metrics to apply while training
        """
        if metrics is None:
            metrics = ['accuracy']
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
            # loss=tf.keras.losses.CategoricalCrossentropy(),
            loss=SparseCategoricalFocalLoss(2),
            metrics=metrics
        )

    def train(self, X_train: tf.Tensor, y_train: tf.Tensor, X_val: tf.Tensor, y_val: tf.Tensor, batch_size=16, epochs=20, save=False, verbose=False):
        """
        This funtion trains the model
        :param X_train: Train Image data
        :param y_train: Train labels
        :param X_val: Validation Image data
        :param y_val: Validation labels
        :param batch_size: Ammount of samples to feed in the network
        :param epochs: Number or epochs
        :param save: Boolean to save the model
        :param verbose: Boolean to see extra data
        :return: Historic of the model
        """
        callbacks = [
            ReduceLROnPlateau(monitor="val_accuracy", patience=3, factor=0.1, verbose=1, min_lr=1e-6),
            EarlyStopping(monitor="val_accuracy", patience=10, verbose=1)
        ]
        if save:
            callbacks.append(ModelCheckpoint(filepath='D:\\model_vgg_tl.h5', monitor="val_accuracy", verbose=1, save_best_only=True))

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

    def predict_per_class(self, X: tf.Tensor, y: tf.Tensor, verbose=False) -> [int]:
        """
        This function predict each image and return the accuracy per class
        :param X: Image data
        :param y: Labels
        :param verbose: Boolean for printing a bar plot
        :return: Accuracy per class
        """
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
