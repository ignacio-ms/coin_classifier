import cv2
import numpy as np
from sklearn.cluster import KMeans

import tensorflow as tf


def imagen_media_color(imagen, etiquetas, centros):
    f_img = imagen.copy()
    etiquetas = etiquetas.reshape(imagen.shape[0], imagen.shape[1])

    for i in range(centros.shape[0]):
        f_img[etiquetas == i, :] = centros[i]

    return f_img


model = tf.keras.models.load_model('c_classifier/models/model_3.h5')
LABEL_DICT = {0: '1c', 1: '1e', 2: '2c', 3: '2e', 4: '5c', 5: '10c', 6: '20c', 7: '50c'}

img = cv2.imread('c_detector/data/test_5.jpg', cv2.IMREAD_COLOR)

alg = KMeans(n_clusters=2, n_init=10).fit(img.reshape(img.shape[1] * img.shape[0], 3))
img_med = imagen_media_color(img, alg.labels_, alg.cluster_centers_)

img_gray = cv2.cvtColor(img_med, cv2.COLOR_BGR2GRAY)
img_gauss = cv2.GaussianBlur(img_gray, (21, 21), cv2.BORDER_DEFAULT)

circles = cv2.HoughCircles(img_gauss, cv2.HOUGH_GRADIENT, 1, 90, param1=50, param2=30)
circles_rounded = np.uint16(np.round(circles))
print(f'{circles_rounded.shape[1]} coins found')

pred = []
for i in circles_rounded[0, :]:
    new_img = img[i[1]-i[2]: i[1]+i[2], i[0]-i[2]: i[0]+i[2]]
    new_img = cv2.resize(new_img, (150, 150))
    new_img = cv2.cvtColor(new_img, cv2.COLOR_BGR2RGB)

    pred.append(int(np.argmax(model.predict(tf.reshape(new_img, (1, 150, 150, 3))))))

label = 0
for i in circles_rounded[0, :]:
    cv2.rectangle(img, (i[0] - i[2] - 10, i[1] - i[2] - 10), (i[0] + i[2] + 10, i[1] + i[2] + 10), (0, 0, 255), 2)
    cv2.putText(img, f'{LABEL_DICT.get(pred[label])}', (i[0] - i[2] - 10, i[1] - i[2] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    label += 1


cv2.imshow('Segmentacion KMeans', np.hstack((img, img_med)))
cv2.waitKey(0)
cv2.destroyAllWindows()
