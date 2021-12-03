import cv2
import numpy as np
import tensorflow as tf


BINS = [0.5, 0.75, 0.9, 2]
COLOR_DICT = {0: (0, 0, 255), 1: (0, 255, 255), 2: (0, 165, 255), 3: (0, 255, 0)}

model = tf.keras.models.load_model('c_classifier/models/model_vgg_2.h5')
LABEL_DICT = {0: '1c', 1: '1e', 2: '2c', 3: '2e', 4: '5c', 5: '10c', 6: '20c', 7: '50c', 8: 'Not Found'}

img = cv2.imread('data/mixed/test_2.jpg', cv2.IMREAD_COLOR)
img = cv2.resize(img, (1209, 906))

img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_gauss = cv2.GaussianBlur(img_gray, (21, 21), cv2.BORDER_DEFAULT)
circles = cv2.HoughCircles(img_gauss, cv2.HOUGH_GRADIENT, 1, 80, param1=50, param2=30)
circles_rounded = np.uint16(np.round(circles))
print(f'{circles_rounded.shape[1]} coins found')

data = np.sort(circles_rounded[0, :, 2])
q3, q1 = np.percentile(data, [75, 25])
iqr = q3 - q1
low_lim = q1 - 3 * iqr
up_lim = q3 + 3 * iqr

pred = []
pred_prob = []
for i in circles_rounded[0, :]:
    if low_lim <= i[2] <= up_lim:
        new_img = img[i[1]-i[2]-10: i[1]+i[2]+10, i[0]-i[2]-10: i[0]+i[2]+10]
        new_img = cv2.resize(new_img, (150, 150))
        new_img = cv2.cvtColor(new_img, cv2.COLOR_BGR2RGB)
        new_img = new_img / 255.0
        new_img = new_img.astype(np.float32)
        pred_prob.append(np.max(model.predict(tf.reshape(new_img, (1, 150, 150, 3)))))
        pred.append(int(np.argmax(model.predict(tf.reshape(new_img, (1, 150, 150, 3))))))

label = 0
colors = [COLOR_DICT.get(i) for i in np.digitize(pred_prob, BINS)]
for i in circles_rounded[0, :]:
    if low_lim <= i[2] <= up_lim:
        cv2.rectangle(img, (i[0]-i[2]-10, i[1]-i[2]-10), (i[0]+i[2]+10, i[1]+i[2]+10), colors[label], 2)
        cv2.putText(img, f'{LABEL_DICT.get(pred[label])}', (i[0] - i[2] - 10, i[1] - i[2] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, colors[label], 2)
        label += 1

cv2.putText(img, '90-100%', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, COLOR_DICT.get(3), 2)
cv2.putText(img, '75-90%', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, COLOR_DICT.get(2), 2)
cv2.putText(img, '50-75%', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.9, COLOR_DICT.get(1), 2)
cv2.putText(img, '0-50%', (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.9, COLOR_DICT.get(0), 2)

cv2.imshow('Detection', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
