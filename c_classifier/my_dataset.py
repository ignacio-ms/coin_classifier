import os

import cv2
import keras.preprocessing.image
import numpy as np
import tensorflow as tf

from helpers import timed


# Numpy dataset (Used for preprocessing)
class MyNpDataset:

    def __init__(self):
        self.IMG_LABELS = ['1c', '1e', '2c', '2e', '5c', '10c', '20c', '50c']

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

        # Reading data as Numpy arrays from disk
        self.data = np.array([self.read_img(path) for path in img_paths])
        self.labels = np.array(labels)

    @staticmethod
    def read_img(img_path):
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = img.astype(np.uint8)
        img = np.array(img)
        return img

    @timed
    def save_data(self):
        for i in range(len(self.data)):
            cv2.imwrite(self.data_paths[i], self.data[i])

    def print_img_by_index(self, index):
        if len(self.data) == 0:
            raise ValueError("Can't print empty dataset\n")

        cv2.imshow(self.labels[index], self.data[index])
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def __str__(self):
        print('----------Dataset----------')
        print(f'No.Examples: {len(self.labels)}')
        print(f'No.Clases: {len(np.unique(self.labels))}')
        for c in np.unique(self.labels):
            print(f'\tClass {c}: {sum(np.where(self.labels == c, 1, 0))}')
        print(f'Image shapes: {self.data[0].shape}')
        return 'NpDataset'


# Tensorflow Tensors dataset (Used for training)
class MyTfDataset:

    def __init__(self):
        self.IMG_LABELS = ['1c', '1e', '2c', '2e', '5c', '10c', '20c', '50c']
        self.LABEL_DICT = {'1c': 0, '1e': 1, '2c': 2, '2e': 3, '5c': 4, '10c': 5, '20c': 6, '50c': 7}

        self.data = []
        self.labels = []
        self.labels_oh = []

    @timed
    def read_data(self, datset_path, augmentation=False):
        # Reading image paths and creating labels
        img_paths, labels = list(), list()

        for file in self.IMG_LABELS:
            f_dir = os.path.join(datset_path, file)
            walk = os.walk(f_dir).__next__()
            for img in walk[2]:
                if img.endswith('.jpg'):
                    img_paths.append(os.path.join(f_dir, img))
                    labels.append(file)

        labels = [self.LABEL_DICT[v] for v in labels]

        # Reading images from disk
        ds = tf.data.Dataset.from_tensor_slices((img_paths, labels))
        ds = ds.map(self.map_img)

        aug = keras.preprocessing.image.ImageDataGenerator(
            rotation_range=180,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            fill_mode='nearest'
        )

        # Saving data
        index = 0
        for X, y in ds:
            self.data.append(X[0])
            self.labels_oh.append(y)
            self.labels.append(labels[index])

            if augmentation:
                aug_iter = aug.flow(X)
                r = 10 if labels[index] == 2 else (4 if labels[index] == 0 else 2)
                for i in range(r):
                    img = next(aug_iter)[0].astype(np.float32)
                    self.data.append(img)
                    self.labels_oh.append(y)
                    self.labels.append(labels[index])

            index += 1

        # Shuffle data
        index = tf.range(start=0, limit=tf.shape(self.data)[0], dtype=tf.int32)
        s_index = tf.random.shuffle(index)

        self.data = tf.gather(self.data, s_index)
        self.labels_oh = tf.gather(self.labels_oh, s_index)
        self.labels = tf.gather(self.labels, s_index)

    @staticmethod
    def read_img(img_path):
        img_path = img_path.decode('utf-8')
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img / 255.0
        img = img.astype(np.float32)
        return img

    def map_img(self, img_path, label):
        img = tf.numpy_function(self.read_img, [img_path], [tf.float32])
        label = tf.one_hot(label, 8, dtype=tf.int32)
        return img, label

    def print_img_by_index(self, index):
        if len(self.data) == 0:
            raise ValueError("Can't print empty dataset\n")

        label = np.array(list(self.LABEL_DICT.keys()))[self.labels[index]]
        img = self.data[index].numpy()
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        cv2.imshow(label, img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def validation_split(self, keep_size=0.8):
        if len(self.data) == 0:
            raise ValueError("Can't split empty dataset\n")

        # No. elements to keep
        no_keep_size = int(len(self.data) * keep_size)

        val = MyTfDataset()
        val.data = self.data[no_keep_size:]
        val.labels = self.labels[no_keep_size:]
        val.labels_oh = self.labels_oh[no_keep_size:]

        self.data = self.data[:no_keep_size]
        self.labels = self.labels[:no_keep_size]
        self.labels_oh = self.labels_oh[:no_keep_size]

        return val

    def __str__(self):
        print('----------Dataset----------')
        print(f'No.Examples: {len(self.labels)}')
        print(f'No.Clases: {len(np.unique(self.labels))}')
        for c in np.unique(self.labels):
            print(f'\tClass {self.IMG_LABELS[c]}: {sum(np.where(self.labels == c, 1, 0))}')
        print(f'Image shapes: {self.data[0].shape}')
        return 'TfDataset'
