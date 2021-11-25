from my_dataset import MyNpDataset
import data_preprocessing as eda

import cv2
import matplotlib.pyplot as plt


p_train = MyNpDataset()
p_train.read_data(datset_path='data/train/')
print(p_train)

index = 200
img_noise = eda.noise_reduction(p_train.data[index], 3)
img_gauss = eda.adjust_gamma(img_noise, 1.5)

cv2.imshow('Original', p_train.data[index])
cv2.imshow('Denoised', img_noise)
cv2.imshow('Gaussian', img_gauss)
cv2.waitKey(0)
cv2.destroyAllWindows()

# # Noise reduction
# for i in range(1, 5):
#     plt.subplot(2, 4, i)
#     img = cv2.cvtColor(p_train.data[i - 1], cv2.COLOR_BGR2RGB)
#     _ = plt.imshow(img)
#
# p_train.data = eda.noise_reduction(p_train.data, Smax=5, threshold=55)
# # p_train.save_data()
#
# for i in range(5, 9):
#     plt.subplot(2, 4, i)
#     img = cv2.cvtColor(p_train.data[i-5], cv2.COLOR_BGR2RGB)
#     _ = plt.imshow(img)
# plt.tight_layout()
# plt.show()
