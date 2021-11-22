import os

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
        """
        Read images from a given forlder of subfolders (Corresponding with the image classes). The function saves
        the images and their respective class into the data and labels attributes of the class. Both are saved as
        tensorflow tensors.
        :param datset_path: Path of the file containing the data subfolders
        :param verbose: Boolean to show extra info of the dataset
        :param shuffle: Boolean to shuffle the dataset
        :return:
        """
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
            print('--------Data Readed--------')
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
    def read_img(img_path, label):
        """
        Load an image from disk as a tensor
        :param img_path: Paht of the image to read
        :param label: Class of the image
        :return: Loaded image and it's class
        """
        img = tf.io.read_file(img_path)
        img = tf.image.decode_image(img, dtype=tf.float32)
        return img, label

    def print_img_by_index(self, index):
        """
        Print an image of the dataset corresponging to a given index
        :param index: Index of the image to show
        :return:
        """
        if index >= len(self.labels):
            raise IndexError("Invalid image index")

        plt.title(self.labels[index].numpy())
        _ = plt.imshow(self.data[index])
        plt.show()


train = MyDataset()
train.read_data(datset_path='data/train/', verbose=False, shuffle=True)
train.print_img_by_index(2000)
