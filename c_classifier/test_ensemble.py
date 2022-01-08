from focal_loss import SparseCategoricalFocalLoss

import tensorflow as tf
import numpy as np
import csv
import cv2
import os


tf.random.set_seed(12345)
np.random.seed(12345)

datset_path = 'data/public_test/'
LABEL_DICT = {0: '10c', 1: '1c', 2: '1e', 3: '20c', 4: '2c', 5: '2e', 6: '50c', 7: '5c'}

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

model_cnn = tf.keras.models.load_model('models/model_alex.h5', custom_objects={'SparseCategoricalFocalLoss': SparseCategoricalFocalLoss(2)})
model_res_net = tf.keras.models.load_model('models/model_res_net.h5', custom_objects={'SparseCategoricalFocalLoss': SparseCategoricalFocalLoss(2)})
model_tl_vgg = tf.keras.models.load_model('models/model_vgg_tl.h5', custom_objects={'SparseCategoricalFocalLoss': SparseCategoricalFocalLoss(2)})

# ----- Ensemble ----- #
models = [model_cnn, model_res_net, model_tl_vgg]

predictions_test = []
proba_test = []

for model in models:
    predictions_test.append(np.argmax(model.predict(test), axis=1))
    proba_test.append(np.max(model.predict(test), axis=1))

predictions_test = np.array(predictions_test)
proba_test = np.array(proba_test)

test_pred = [predictions_test[np.argmax(proba_test[:, i], axis=0), i] for i in range(predictions_test.shape[1])]
test_labels = [LABEL_DICT.get(pred) for pred in test_pred]

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
        csvwriter.writerow([test_image_ids[i], test_pred[i]])
