import cv2
import numpy as np

from helpers import timed
from numba import jit, cuda


@timed
def noise_reduction(data, Smax=5, threshold=55):
    if len(data.shape) == 3:
        return adaptative(data, Smax, threshold)

    for i in range(len(data)):
        data[i] = adaptative(data[i], Smax, threshold)

    return data


@jit(forceobj=True)
def adaptative(img, Smax, t):
    ap_max = Smax // 2
    a_imagen = cv2.copyMakeBorder(img, ap_max, ap_max, ap_max, ap_max, cv2.BORDER_REPLICATE)
    f_img = img.copy()

    f, c = img.shape[:2]
    for i in range(f):
        for j in range(c):
            ap_aux = 1
            e = img[i, j, :]

            while ap_aux <= ap_max and len(e) != 0 and (check_max_min(e, t)):
                e = np.median(a_imagen[i - ap_aux:i + ap_aux + 1, j - 1:j + ap_aux + 1, :], axis=(0, 1))
                ap_aux += 1
            f_img[i, j, :] = e

    return f_img


@jit
def check_max_min(arr, t):
    if (0 <= arr[0] <= t and 0 <= arr[1] <= t and 0 <= arr[2] <= t)\
            or ((255 - t) <= arr[0] <= 255 and (255 - t) <= arr[1] <= 255 and (255 - t) <= arr[2] <= 255):
        return True
    return False
