from nets.cnn import CNN
from nets.res_cnn import ResNet
from nets.tl_cnn import TransferVGG

from focal_loss import SparseCategoricalFocalLoss

from my_dataset import MyTfDataset
import tensorflow as tf
import numpy as np


tf.random.set_seed(12345)
np.random.seed(12345)

train = MyTfDataset()
train.read_data(datset_path='data/train/', augmentation=True)
val = train.validation_split(0.7)

print(f'Train {train}')
print(f'Validation {val}')

# # ----- CNN Training and Evaluation ----- #
model_cnn = CNN()
model_cnn.compile()
model_cnn.train(train.data, train.labels, val.data, val.labels, batch_size=8, epochs=100, save=True, verbose=True)

m_cnn = tf.keras.models.load_model('D:\\model_alex.h5', custom_objects={'SparseCategoricalFocalLoss': SparseCategoricalFocalLoss(2)})
m_cnn.evaluate(val.data, val.labels, batch_size=8, verbose=1)

# # ----- ResNet Training and Evaluation ----- #
model_res_net = ResNet(num_blocks_list=[3, 4, 6, 3], num_filters=[64, 64, 256], kernel_size=3)
model_res_net.build_net()
model_res_net.compile()
model_res_net.train(train.data, train.labels, val.data, val.labels, batch_size=8, epochs=100, save=True, verbose=True)

m_res = tf.keras.models.load_model('D:\\model_res_net.h5', custom_objects={'SparseCategoricalFocalLoss': SparseCategoricalFocalLoss(2)})
m_res.evaluate(val.data, val.labels, batch_size=8, verbose=1)

# ----- TransferVGG Training and Evaluation ----- #
model_tl_vgg = TransferVGG()
model_tl_vgg.build_top()
model_tl_vgg.compile()
model_tl_vgg.train(train.data, train.labels, val.data, val.labels, batch_size=8, epochs=100, save=True, verbose=True)

m_vgg = tf.keras.models.load_model('D:\\model_vgg_tl.h5', custom_objects={'SparseCategoricalFocalLoss': SparseCategoricalFocalLoss(2)})
m_vgg.evaluate(val.data, val.labels, batch_size=8, verbose=1)

# ----- Ensemble ----- #
models = [m_cnn, m_res, m_vgg]

predictions_val = []
proba_val = []

for model in models:
    predictions_val.append(np.argmax(model.predict(val.data), axis=1))
    proba_val.append(np.max(model.predict(val.data), axis=1))

predictions_val = np.array(predictions_val)
proba_val = np.array(proba_val)

labels_val = [predictions_val[np.argmax(proba_val[:, i]), i] for i in range(predictions_val.shape[1])]
acc_val = np.sum(np.where(labels_val == val.labels, 1, 0)) / predictions_val.shape[1]
print(f'Accuracy Val: {acc_val}')
