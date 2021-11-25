from my_dataset import MyTfDataset

import time

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

import matplotlib.pyplot as plt

tf.random.set_seed(12345)

train = MyTfDataset()
train.read_data(datset_path='data/train/', augmentation=True)
val = train.validation_split()

print(f'Train {train}')
print(f'Validation {val}')

# data_augmentation = Sequential(
#     [
#         layers.experimental.preprocessing.RandomFlip(mode="horizontal_and_vertical"),
#         layers.experimental.preprocessing.RandomZoom(0.2),
#         layers.experimental.preprocessing.RandomContrast(factor=0.1)
#     ]
# )

# Create Model
# LeNet-5 CNN Architecture
model = Sequential([
    # data_augmentation,
    layers.Conv2D(6, 5, padding='same', activation='relu', input_shape=(150, 150, 3)),
    layers.AveragePooling2D(),
    layers.Conv2D(16, 5, padding='same', activation='relu'),
    layers.AveragePooling2D(),
    layers.Flatten(),
    layers.Dense(120, activation='relu'),
    layers.Dense(84, activation='relu'),
    layers.Dense(8, activation='softmax')  # No.Classes
])

# Compile Model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss=tf.keras.losses.CategoricalCrossentropy(),
    metrics=['accuracy']
)

# Train Model
epochs = 50
strat_time = time.time()
history = model.fit(
    train.data,
    train.labels_oh,
    validation_data=(val.data, val.labels_oh),
    epochs=epochs
)
print(f'Model took {time.time() - strat_time}[s] to train.')
model.summary()

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
