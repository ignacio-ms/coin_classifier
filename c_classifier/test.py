from my_dataset import MyTfDataset

import cv2
import numpy as np
import tensorflow as tf


tf.random.set_seed(12345)

train = MyTfDataset()
train.read_data(datset_path='data/train/', augmentation=True)
val = train.validation_split()

model = tf.keras.models.load_model('models/model_3.h5')
pred = np.argmax(model.predict(val.data), axis=1)

acc = np.sum(np.where(pred == val.labels.numpy(), 1, 0)) / len(pred)

acc_per_class = []
for c in np.unique(val.labels):
    c_pred = np.sum(np.where(pred[val.labels == c] == val.labels[val.labels == c], 1, 0))
    acc_per_class.append(c_pred / np.sum(np.where(val.labels == c, 1, 0)))

print(f'Acc: {acc}')
print(f'Acc per class: {acc_per_class}')


LABEL_DICT = {0: '1c', 1: '1e', 2: '2c', 3: '2e', 4: '5c', 5: '10c', 6: '20c', 7: '50c'}
index = np.random.randint(0, 500, size=10)
for i in index:
    predict = LABEL_DICT.get(int(np.argmax(model.predict(tf.reshape(val.data[i], (1, 150, 150, 3))))))
    img = cv2.cvtColor(val.data[i].numpy(), cv2.COLOR_RGB2BGR)
    cv2.imshow(f'{LABEL_DICT.get(val.labels[i].numpy())} - {predict}', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
