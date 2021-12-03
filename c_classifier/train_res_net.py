import tensorflow as tf
from my_dataset import MyTfDataset
from res_net import ResNet


tf.random.set_seed(12345)

train = MyTfDataset()
train.read_data(datset_path='data/train/', augmentation=True)
val = train.validation_split()

print(f'Train {train}')
print(f'Validation {val}')

model = ResNet(num_blocks_list=[2, 5, 5, 2], num_filters=[64, 64, 128], kernel_size=3)
model.build_net()
model.compile()
model.train(train.data, train.labels_oh, val.data, val.labels_oh, batch_size=8, epochs=20, save=True, verbose=True)

pred = model.predict_per_class(val.data, val.labels, verbose=True)
