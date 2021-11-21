import os
import numpy as np
import tensorflow as tf


class MyDataset:

    def __init__(self):
        self.IMG_SIZE = (150, 150)
        self.IMG_LABELS = ['1c', '1e', '2c', '2e', '5c', '10c', '20c', '50c']

        self.data = None

    def read_data(self, datset_path, verbose=False, GPU=False):
        # Reading image paths and creating labels
        img_paths, labels = list(), list()

        for file in self.IMG_LABELS:
            f_dir = os.path.join(datset_path, file)
            walk = os.walk(f_dir).__next__()
            for img in walk[2]:
                if img.endswith('.jpg'):
                    img_paths.append(os.path.join(f_dir, img))
                    labels.append(file)

        device_name = '/cpu:0'
        # Cheking aviable GPU if requested
        if GPU:
            if tf.test.gpu_device_name() != '/device:GPU:0':
                raise SystemError('GPU device not found...Continue with CPU')
            else:
                device_name = tf.test.gpu_device_name()

        # Convert paths to tensors and read images from disk
        with tf.device(device_name):
            img_paths = tf.convert_to_tensor(img_paths, dtype=tf.string)
            labels = tf.convert_to_tensor(labels, dtype=tf.string)
            
            data = tf.data.Dataset.from_tensor_slices((img_paths, labels))
            data = data.map(read_img)
            self.data = data
            
        if verbose:
            print('--------Images--------')
            print(f'Total valid image paths: {len(img_paths)}')
            print(f'{len(np.unique(labels))} diferent classes: {np.unique(labels)}')
            
    def read_img(img_path, label):
        img = tf.io.read_file(img_path, label)
        img = tf.image.decode_image(img, dtype=tf.float32)
        return img, label


dataset = MyDataset()
dataset.read_data(datset_path='data/train/', verbose=True, GPU=True)
