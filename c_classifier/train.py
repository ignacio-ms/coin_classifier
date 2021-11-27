from my_dataset import MyTfDataset
from cnn_architectures import CNN

import tensorflow as tf


tf.random.set_seed(12345)

train = MyTfDataset()
train.read_data(datset_path='data/train/', augmentation=True)
val = train.validation_split()

print(f'Train {train}')
print(f'Validation {val}')

# le_net = CNN(architecture='LeNet')
# le_net.compile()
# le_net.train(train.data, train.labels_oh, val.data, val.labels_oh, epochs=25, verbose=True)

# le_net = CNN(architecture='Genetic')
# le_net.compile()
# le_net.train(train.data, train.labels_oh, val.data, val.labels_oh, epochs=15, verbose=True)

le_net = CNN(architecture='GeneticV2')
le_net.compile()
le_net.train(train.data, train.labels_oh, val.data, val.labels_oh, epochs=15, verbose=True)
