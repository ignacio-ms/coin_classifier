import cv2
import numpy as np
from skimage.segmentation import slic


def crea_imagen_final(imagen, labels, K):

    f_img = imagen.copy()

    for i in range(K):
        f_img[labels == i, :] = np.mean(imagen[labels == i, :], axis=0)

    return f_img


img = cv2.imread('data/test_1.jpg')

K = 1000
super_p = slic(img.copy(), n_segments=K, start_label=0)
img_sp = crea_imagen_final(img.copy(), super_p, K)

img_gray = cv2.cvtColor(img_sp, cv2.COLOR_BGR2GRAY)
img_gauss = cv2.GaussianBlur(img_gray, (21, 21), cv2.BORDER_DEFAULT)

circles = cv2.HoughCircles(img_gauss, cv2.HOUGH_GRADIENT, 1, 100, param1=50, param2=30)
circles_rounded = np.uint16(np.round(circles))
print(f'{circles_rounded.shape[1]} coins found')

for i in circles_rounded[0, :]:
    cv2.circle(img, (i[0], i[1]), i[2], (0, 255, 0), 2)
    cv2.circle(img, (i[0], i[1]), 2, (255, 0, 0), 3)

    # cv2.rectangle(img, (i[0] - i[2] - 10, i[1] - i[2] - 10), (i[0] + i[2] + 10, i[1] + i[2] + 10), (0, 0, 255), 2)

cv2.imshow('Superpixels_boundaries', np.hstack((img, img_sp)))
cv2.waitKey(0)
cv2.destroyAllWindows()
