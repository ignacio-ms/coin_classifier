import os
import cv2
import numpy as np
import tensorflow as tf


tf.random.set_seed(12345)
datset_path = 'data/public_test/'

test = []
walk = os.walk(datset_path).__next__()
for img in walk[2]:
    if img.endswith('.jpg'):
        img = cv2.imread(os.path.join(datset_path, img), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img / 255.0
        img = img.astype(np.float32)
        test.append(img)

test = np.array(test)
model = tf.keras.models.load_model('models/model_3.h5')
pred = np.argmax(model.predict(test), axis=1)

LABEL_DICT = {0: '1c', 1: '1e', 2: '2c', 3: '2e', 4: '5c', 5: '10c', 6: '20c', 7: '50c'}
index = np.random.randint(0, 100, size=15)
for i in index:
    predict = LABEL_DICT.get(int(np.argmax(model.predict(tf.reshape(test[i], (1, 150, 150, 3))))))
    img = cv2.cvtColor(test[i], cv2.COLOR_RGB2BGR)
    cv2.imshow(f'{predict}', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
