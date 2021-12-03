from my_dataset import MyTfDataset
from cnn import CNN

import numpy as np
import tensorflow as tf


tf.random.set_seed(12345)
np.random.seed(12345)

train = MyTfDataset()
train.read_data(datset_path='data/train/', augmentation=True)
val = train.validation_split()

print(f'Train {train}')
print(f'Validation {val}')

arch = [96, 256, 512, 1024, 1024, 11, 5, 3, 3, 3]

model = CNN(arch[:5], arch[5:])
model.compile()
model.train(train.data, train.labels_oh, val.data, val.labels_oh, batch_size=8, epochs=25, save=True, verbose=True)

pred = model.predict_per_class(val.data, val.labels, verbose=True)
