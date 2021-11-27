from my_dataset import MyNpDataset
import data_preprocessing as eda

import cv2
import matplotlib.pyplot as plt


ds = MyNpDataset()
ds.read_data(datset_path='data/train/')
print(ds)

# Noise reduction
for i in range(1, 5):
    plt.subplot(2, 4, i)
    img = cv2.cvtColor(ds.data[i - 1], cv2.COLOR_BGR2RGB)
    _ = plt.imshow(img)

ds.data = eda.noise_reduction(ds.data, 3)
ds.data = eda.brightness_correction(ds.data, method='hsv')
ds.save_data()

for i in range(5, 9):
    plt.subplot(2, 4, i)
    img = cv2.cvtColor(ds.data[i-5], cv2.COLOR_BGR2RGB)
    _ = plt.imshow(img)
plt.tight_layout()
plt.show()
