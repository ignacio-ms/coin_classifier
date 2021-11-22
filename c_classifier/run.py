import numpy as np

from my_dataset import MyDataset
import data_preprocessing as eda

import tensorflow as tf


train = MyDataset()
train.read_data(datset_path='data/train/', verbose=False, shuffle=True)

print(train.data[0])
train.data[0] = eda.adaptative(train.data[0], 9)
