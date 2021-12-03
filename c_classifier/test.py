import os
import cv2
import csv
import numpy as np
import tensorflow as tf


tf.random.set_seed(12345)
datset_path = 'data/public_test/'
LABEL_DICT = {0: '1c', 1: '1e', 2: '2c', 3: '2e', 4: '5c', 5: '10c', 6: '20c', 7: '50c'}
LABEL_DICT_KAGGLE = {0: 1, 1: 2, 2: 4, 3: 5, 4: 7, 5: 0, 6: 3, 7: 6}

test = []
test_image_ids = []
walk = os.walk(datset_path).__next__()
for img in walk[2]:
    if img.endswith('.jpg'):
        test_image_ids.append(os.path.join(img))

        img = cv2.imread(os.path.join(datset_path, img), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img / 255.0
        img = img.astype(np.float32)
        test.append(img)

test = np.array(test)
model = tf.keras.models.load_model('models/model_res_net.h5')
test_pred = np.argmax(model.predict(test), axis=1)
test_labels = [LABEL_DICT.get(pred) for pred in test_pred]

index = np.random.randint(0, 965, size=20)
for i in index:
    predict = LABEL_DICT.get(test_pred[i])
    img = cv2.cvtColor(test[i], cv2.COLOR_RGB2BGR)
    cv2.imshow(f'{predict} - {test_image_ids[i]}', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

fields = ["Id", "Expected"]
filename = "results.csv"

# writing to csv file
with open(filename, 'w', newline="") as csvfile:
    # creating a csv writer object
    csvwriter = csv.writer(csvfile, delimiter=',')

    # writing the fields
    csvwriter.writerow(fields)

    # writing the data rows
    for i in range(len(test_pred)):
        csvwriter.writerow([test_image_ids[i], LABEL_DICT_KAGGLE.get(test_pred[i])])
