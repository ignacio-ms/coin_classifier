from focal_loss import SparseCategoricalFocalLoss
import tensorflow as tf

import numpy as np
import cv2


class Detector:

    def __init__(self):
        self.ensemble = None

        self.BINS = [0.5, 0.75, 0.9, 1.1]
        self.COLOR_DICT = {0: (0, 0, 255), 1: (0, 255, 255), 2: (0, 165, 255), 3: (0, 255, 0)}

        self.LABEL_DICT = {0: '10c', 1: '1c', 2: '1e', 3: '20c', 4: '2c', 5: '2e', 6: '50c', 7: '5c'}
        self.VALUE_DICT = {0: 0.1, 1: 0.01, 2: 1, 3: 0.2, 4: 0.02, 5: 2, 6: 0.5, 7: 0.05}

    def load_ensemble(self):
        """
        Load the three CNN models and combine them on an array.
        """
        model_cnn = tf.keras.models.load_model('c_classifier/models/model_alex.h5', custom_objects={'SparseCategoricalFocalLoss': SparseCategoricalFocalLoss(2)})
        model_res_net = tf.keras.models.load_model('c_classifier/models/model_res_net.h5', custom_objects={'SparseCategoricalFocalLoss': SparseCategoricalFocalLoss(2)})
        model_tl_vgg = tf.keras.models.load_model('c_classifier/models/model_vgg_tl.h5', custom_objects={'SparseCategoricalFocalLoss': SparseCategoricalFocalLoss(2)})

        self.ensemble = [model_cnn, model_res_net, model_tl_vgg]

    @staticmethod
    def detect(img: np.ndarray) -> np.ndarray:
        """
        This function detects circles in an image using the Hough Circle Transform.
        :param img: Image to detect circles
        :return: Array of circles
        """
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_gauss = cv2.GaussianBlur(img_gray, (21, 21), cv2.BORDER_DEFAULT)
        circles = cv2.HoughCircles(img_gauss, cv2.HOUGH_GRADIENT, 1, 80, param1=50, param2=30)

        return np.uint16(np.round(circles))

    @staticmethod
    def IQR(circles: np.ndarray) -> np.ndarray:
        """
        This function applies the Interquirtile Range Outliner detection method.
        :param circles: Array of circles
        :return: Preprocessed array or circles
        """
        data = np.sort(circles[0, :, 2])
        q3, q1 = np.percentile(data, [75, 25])
        iqr = q3 - q1
        low_lim = q1 - 3 * iqr
        up_lim = q3 + 3 * iqr

        coins = np.array([circles[0, i] for i in range(circles.shape[1]) if low_lim <= circles[0, i, 2] <= up_lim])
        return coins

    def classify(self, img: np.ndarray, coins: np.ndarray) -> (np.ndarray, np.ndarray):
        """
        This function takes the fraction of the image corresponding to each detected coin and classify it.
        :param img: Image to detect coins
        :param coins: Coin array
        :return: Arrays of predicted labels and its corresponding propability
        """
        pred = []
        pred_prob = []
        for i in coins:
            new_img = img[i[1] - i[2] - 10: i[1] + i[2] + 10, i[0] - i[2] - 10: i[0] + i[2] + 10]
            new_img = cv2.resize(new_img, (150, 150))
            new_img = cv2.cvtColor(new_img, cv2.COLOR_BGR2RGB)
            new_img = new_img / 255.0
            new_img = new_img.astype(np.float32)

            # Voting
            c_pred = []
            c_pred_prob = []
            for model in self.ensemble:
                c_pred_prob.append(np.max(model.predict(tf.reshape(new_img, (1, 150, 150, 3)))))
                c_pred.append(int(np.argmax(model.predict(tf.reshape(new_img, (1, 150, 150, 3))))))

            pred.append(c_pred[np.argmax(c_pred_prob)])
            pred_prob.append(np.max(c_pred_prob))

        return pred, pred_prob

    def print_coins(self, img: np.ndarray, coins: np.ndarray, pred: [int], pred_prob: [int]):
        """
        This function draws a box arround the detected coins with its predicted label.
        Each box colored indicating the confidence of the prediction.
        Additionaly this function counts the total ammount of money in the image (In euros)
        :param img: Image to modify
        :param coins: Cloins to draw
        :param pred: Coin labels
        :param pred_prob: Coin labels confidence
        """
        label = 0
        colors = [self.COLOR_DICT.get(i) for i in np.digitize(pred_prob, self.BINS)]
        for i in coins:
            cv2.rectangle(img, (i[0] - i[2] - 10, i[1] - i[2] - 10), (i[0] + i[2] + 10, i[1] + i[2] + 10), colors[label], 2)
            cv2.putText(img, f'{self.LABEL_DICT.get(pred[label])}', (i[0] - i[2] - 10, i[1] - i[2] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, colors[label], 2)
            label += 1

        coin_count = np.round(np.sum([self.VALUE_DICT.get(pred[i]) for i in range(len(pred))]), 2)

        cv2.putText(img, f'{len(coins)} coins found', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.putText(img, f'Total count: {coin_count} euros', (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        cv2.putText(img, 'Confidence:', (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.putText(img, '90-100%', (10, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.COLOR_DICT.get(3), 2)
        cv2.putText(img, '75-90%', (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.COLOR_DICT.get(2), 2)
        cv2.putText(img, '50-75%', (10, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.COLOR_DICT.get(1), 2)
        cv2.putText(img, '0-50%', (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.COLOR_DICT.get(0), 2)

        cv2.imshow('Detection', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
