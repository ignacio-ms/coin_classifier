import os
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf

from helpers import timed


class MyDataset:

    def __init__(self, ds_type='np'):
        if ds_type not in ['np', 'tf']:
            raise TypeError('Dataset type must be "np" or "tf".')
        self.ds_type = ds_type

        self.IMG_SIZE = (150, 150)
        self.IMG_LABELS = ['1c', '1e', '2c', '2e', '5c', '10c', '20c', '50c']
        self.label_dict = {'1c': 0, '1e': 1, '2c': 2, '2e': 3, '5c': 4, '10c': 5, '20c': 6, '50c': 7}

        self.data = []
        self.labels = []
        self.data_paths = []

    @timed
    def read_data(self, datset_path):
        # Reading image paths and creating labels
        img_paths, labels = list(), list()

        for file in self.IMG_LABELS:
            f_dir = os.path.join(datset_path, file)
            walk = os.walk(f_dir).__next__()
            for img in walk[2]:
                if img.endswith('.jpg'):
                    img_paths.append(os.path.join(f_dir, img))
                    labels.append(file)

        self.data_paths = np.array(img_paths)
        labels = [self.label_dict[v] for v in labels]

        # Reading data as EigerTensors (Used for training)
        if self.ds_type == 'tf':
            # Reading images from disk
            ds = tf.data.Dataset.from_tensor_slices((img_paths, labels))
            ds = ds.map(self.map_img)

            # Saving data
            for X, y in ds:
                self.data.append(X[0])
                self.labels.append(y)

            # Shuffle data
            index = tf.range(start=0, limit=tf.shape(self.data)[0], dtype=tf.int32)
            s_index = tf.random.shuffle(index)

            self.data = tf.gather(self.data, s_index)
            self.labels = tf.gather(self.labels, s_index)

        # Reading data as Numpy arrays (Used for preprocessing)
        elif self.ds_type == 'np':
            # Saving and Reading images from disk
            self.data = np.array([self.read_img_np(path) for path in img_paths])
            self.labels = np.array(labels)

    @staticmethod
    def read_img_tf(img_path):
        img_path = img_path.decode('utf-8')
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img / 255.0
        img = img.astype(np.float32)
        return img

    @staticmethod
    def read_img_np(img_path):
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = img.astype(np.uint8)
        img = np.array(img)
        return img

    def map_img(self, img_path, label):
        img = tf.numpy_function(self.read_img_tf, [img_path], [tf.float32])
        label = tf.one_hot(label, 8, dtype=tf.int32)
        return img, label

    @timed
    def save_data(self):
        if self.ds_type != 'np':
            raise TypeError('Only np type dataset can be saved in disk')

        for i in range(len(self.data)):
            cv2.imwrite(self.data_paths[i], self.data[i])

    def print_img_by_index(self, index, pause=True):
        if len(self.data) == 0:
            raise ValueError("Can't print empty dataset\n")

        label = self.labels[index].numpy().decode('UTF-8') if self.ds_type == 'tf' else self.labels[index]
        img = self.data[index].numpy() if self.ds_type == 'tf' else self.data[index]
        if self.ds_type == 'tf':
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        cv2.imshow(label, img)
        if pause:
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    def validation_split(self, keep_size=0.8):
        if len(self.data) == 0:
            raise ValueError("Can't split empty dataset\n")

        # No. elements to keep
        no_keep_size = int(len(self.data) * keep_size)

        val = MyDataset()
        val.data = self.data[no_keep_size:]
        val.labels = self.labels[no_keep_size:]

        self.data = self.data[:no_keep_size]
        self.labels = self.labels[:no_keep_size]

        return val
