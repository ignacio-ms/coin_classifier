from my_dataset import MyDataset
import data_preprocessing as eda

import cv2
import matplotlib.pyplot as plt


p_train = MyDataset(ds_type='np')
p_train.read_data(datset_path='data/train/')

# Noise reduction
for i in range(1, 5):
    plt.subplot(2, 4, i)
    img = cv2.cvtColor(p_train.data[i - 1], cv2.COLOR_BGR2RGB)
    _ = plt.imshow(img)

p_train.data = eda.noise_reduction(p_train.data, Smax=11, threshold=40)
p_train.save_data()

for i in range(5, 9):
    plt.subplot(2, 4, i)
    img = cv2.cvtColor(p_train.data[i-5], cv2.COLOR_BGR2RGB)
    _ = plt.imshow(img)
plt.tight_layout()
plt.show()
