import cv2
import numpy as np
from sklearn.cluster import KMeans


def imagen_media_color(imagen, etiquetas, centros):
    f_img = imagen.copy()
    etiquetas = etiquetas.reshape(imagen.shape[0], imagen.shape[1])

    for i in range(centros.shape[0]):
        f_img[etiquetas == i, :] = centros[i]

    return f_img


img = cv2.imread('data/test_2.jpg', cv2.IMREAD_COLOR)

alg = KMeans(n_clusters=2, n_init=10).fit(img.reshape(img.shape[1] * img.shape[0], 3))
img_med = imagen_media_color(img, alg.labels_, alg.cluster_centers_)

img_gray = cv2.cvtColor(img_med, cv2.COLOR_BGR2GRAY)
img_gauss = cv2.GaussianBlur(img_gray, (21, 21), cv2.BORDER_DEFAULT)

circles = cv2.HoughCircles(img_gauss, cv2.HOUGH_GRADIENT, 1, 90, param1=50, param2=30)
circles_rounded = np.uint16(np.round(circles))
print(f'{circles_rounded.shape[1]} coins found')

for i in circles_rounded[0, :]:
    cv2.circle(img, (i[0], i[1]), i[2], (0, 255, 0), 2)
    cv2.circle(img, (i[0], i[1]), 2, (255, 0, 0), 3)

    # cv2.rectangle(img, (i[0] - i[2] - 10, i[1] - i[2] - 10), (i[0] + i[2] + 10, i[1] + i[2] + 10), (0, 0, 255), 2)

cv2.imshow('Segmentacion KMeans', np.hstack((img, img_med)))
cv2.waitKey(0)
cv2.destroyAllWindows()
