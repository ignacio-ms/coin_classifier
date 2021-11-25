import cv2
import numpy as np

from helpers import timed
from numba import jit


@timed
def noise_reduction(data, Smax=5, threshold=55, adapt=False):
    if len(data.shape) == 3:
        if adapt:
            return adaptative(data, Smax, threshold)
        return median_filter(data, Smax)

    for i in range(len(data)):
        if adapt:
            data[i] = adaptative(data[i], Smax, threshold)
        else:
            data[i] = median_filter(data[i], Smax)

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


@jit(forceobj=True)
def median_filter(img, S):
    AP = S // 2
    a_imagen = cv2.copyMakeBorder(img, AP, AP, AP, AP, cv2.BORDER_REPLICATE)
    f_img = img.copy()

    f, c = img.shape[:2]
    for i in range(1, f+1):
        for j in range(1, c+1):
            f_img[i-1, j-1] = np.median(a_imagen[i - AP:i + AP + 1, j - 1:j + AP + 1], axis=(0, 1))

    return f_img


@jit
def check_max_min(arr, t):
    if (0 <= arr[0] <= t and 0 <= arr[1] <= t and 0 <= arr[2] <= t) \
            or ((255 - t) <= arr[0] <= 255 and (255 - t) <= arr[1] <= 255 and (255 - t) <= arr[2] <= 255):
        return True
    return False


def adjust_gamma(image, gamma=1.0):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)
