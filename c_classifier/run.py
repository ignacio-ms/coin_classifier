import numpy as np

from my_dataset import MyDataset
import data_preprocessing as eda

import tensorflow as tf


train = MyDataset()
train.read_data(datset_path='data/train/', verbose=False, shuffle=True)

train.data[0].assign(eda.adaptative(train.data[0]))
