import os

import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from helpers import timed


class MyDataset:

    def __init__(self):
        self.IMG_SIZE = (150, 150)
        self.IMG_LABELS = ['1c', '1e', '2c', '2e', '5c', '10c', '20c', '50c']

        self.data = []
        self.labels = []

    @timed
    def read_data(self, datset_path, verbose=False, shuffle=False):
        # Reading image paths and creating labels
        img_paths, labels = list(), list()

        for file in self.IMG_LABELS:
            f_dir = os.path.join(datset_path, file)
            walk = os.walk(f_dir).__next__()
            for img in walk[2]:
                if img.endswith('.jpg'):
                    img_paths.append(os.path.join(f_dir, img))
                    labels.append(file)

        # Reading images from disk
        ds = tf.data.Dataset.from_tensor_slices((img_paths, labels))
        ds = ds.map(self.read_img)

        # Saving data
        for X, y in ds:
            self.data.append(X)
            self.labels.append(y)

        if shuffle:
            index = tf.range(start=0, limit=tf.shape(self.data)[0], dtype=tf.int32)
            s_index = tf.random.shuffle(index)

            self.data = tf.gather(self.data, s_index)
            self.labels = tf.gather(self.labels, s_index)
            
        if verbose:
            # General
            print('----------Data Readed----------')
            print(f'Total valid image paths: {len(img_paths)}')
            print(f'{len(np.unique(labels))} diferent classes: {np.unique(labels)}')

            # Number of images per class
            no_classes = {c: np.sum(np.where(np.array(labels) == c, 1, 0)) for c in np.unique(labels)}
            plt.bar(no_classes.keys(), no_classes.values())
            plt.title("Number of images by class")
            plt.xlabel('Class Name')
            plt.ylabel('# Images')
            plt.show()

    @staticmethod
    def parse_opencv(img_path):
        img_path = img_path.decode('utf-8')
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = img / 255.0
        img = img.astype(np.float32)
        return img

    def read_img(self, img_path, label):
        img = tf.numpy_function(self.parse_opencv, [img_path], [tf.float32])
        return img, label

    def print_img_by_index(self, index):
        label = self.labels[index].numpy().decode('UTF-8')
        img = self.data[index].numpy()
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        cv2.imshow(label, img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def validation_split(self, keep_size=0.8, verbose=False):
        if len(self.data) == 0:
            raise ValueError("Can't split empty dataset\n")

        # No. elements to keep
        no_keep_size = int(len(self.data) * keep_size)

        val = MyDataset()
        val.data = self.data[no_keep_size:]
        val.labels = self.labels[no_keep_size:]

        self.data = self.data[:no_keep_size]
        self.labels = self.labels[:no_keep_size]

        if verbose:
            print('--------Validation split--------')
            print(f'Splitted data from size {len(self.data) + len(val.data)} to sizes')
            print(f'\t{len(self.data)} of keeping data.')
            print(f'\t{len(val.data)} of validation data.')

            # New distribution
            no_classes = {c: np.sum(np.where(np.array(self.labels) == c, 1, 0)) for c in np.unique(self.labels)}
            plt.subplot(2, 1, 1)
            plt.bar(no_classes.keys(), no_classes.values())
            plt.title("Number of images by class in first split")
            plt.xlabel('Class Name')
            plt.ylabel('# Images')

            no_classes = {c: np.sum(np.where(np.array(val.labels) == c, 1, 0)) for c in np.unique(val.labels)}
            plt.subplot(2, 1, 2)
            plt.bar(no_classes.keys(), no_classes.values())
            plt.title("Number of images by class in secons split")
            plt.xlabel('Class Name')
            plt.ylabel('# Images')

            plt.tight_layout()
            plt.show()

        return val
