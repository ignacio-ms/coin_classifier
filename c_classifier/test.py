from my_dataset import MyTfDataset
from cnn import CNN

import numpy as np
import tensorflow as tf


tf.random.set_seed(12345)
np.random.seed(12345)

train = MyTfDataset()
train.read_data(datset_path='data/train/', augmentation=True)
val = train.validation_split()

model = tf.keras.models.load_model('models/model_3.h5')
pred = model.predict(val.data)
pred = np.argmax(pred, axis=1)

prob_per_class = []
for c in np.unique(val.labels):
    c_pred = np.sum(np.where(pred[val.labels == c] == val.labels[val.labels == c], 1, 0))
    prob_per_class.append(c_pred / np.sum(np.where(val.labels == c, 1, 0)))

print(prob_per_class)
