from tensorflow.keras.layers import (
    Input, Conv2D, ReLU,
    BatchNormalization, Add, AveragePooling2D,
    Flatten, Dense
)
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import Model

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

from helpers import timed


class ResNet:

    def __init__(self, num_blocks_list=None, num_filters=None, kernel_size=3):
        if num_filters is None:
            num_filters = [64, 64, 128]
        if num_blocks_list is None:
            num_blocks_list = [2, 5, 5, 2]

        self.num_blocks_list = num_blocks_list
        self.num_filters = num_filters
        self.kernel_size = kernel_size

        self.input_shape = (150, 150, 3)
        self.model = None

    def build_net(self):

        inputs = Input(shape=self.input_shape)

        t = BatchNormalization()(inputs)
        t = Conv2D(filters=self.num_filters[0], kernel_size=self.kernel_size, strides=1, padding="same")(t)
        t = BatchNormalization()(t)
        t = ReLU()(t)

        for i in range(len(self.num_blocks_list)):
            num_blocks = self.num_blocks_list[i]
            for j in range(num_blocks):
                t = self.residual_block(t, identity=(j != 0), filters=self.num_filters, kernel_size=self.kernel_size)
            self.num_filters *= 2

        t = AveragePooling2D(4)(t)
        t = Flatten()(t)
        outputs = Dense(8, activation='softmax')(t)

        self.model = Model(inputs, outputs)

    @staticmethod
    def residual_block(x, identity, filters, kernel_size=3):

        x_skip = x

        x = Conv2D(kernel_size=1, strides=(1 if identity else 2), filters=filters[0], padding="same")(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)

        x = Conv2D(kernel_size=kernel_size, strides=1, filters=filters[1], padding="same")(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)

        x = Conv2D(kernel_size=1, strides=1, filters=filters[2], padding="same")(x)
        x = BatchNormalization()(x)

        if not identity:
            x_skip = Conv2D(kernel_size=1, strides=2, filters=filters[2], padding="same")(x_skip)
            x_skip = BatchNormalization()(x_skip)

        out = Add()([x, x_skip])
        out = ReLU()(out)
        return out

    def compile(self, lr=1e-4, metrics=None):
        if metrics is None:
            metrics = ['accuracy']
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=metrics
        )

    @timed
    def train(self, X_train, y_train, X_val, y_val, batch_size=16, epochs=20, save=False, verbose=False):
        callbacks = [
            ReduceLROnPlateau(monitor="val_accuracy", patience=3, factor=0.1, verbose=1, min_lr=1e-6),
            EarlyStopping(monitor="val_accuracy", patience=5, verbose=1)
        ]
        if save:
            callbacks.append(ModelCheckpoint(filepath='D:\\model_res_net.h5', monitor="val_accuracy", verbose=1, save_best_only=True))

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
