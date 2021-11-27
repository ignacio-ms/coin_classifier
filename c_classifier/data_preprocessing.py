import cv2
import math
import numpy as np

from helpers import timed
from numba import jit

# ----- Noise reduction ----- #
@timed
def noise_reduction(data, Smax=3, threshold=55, adapt=False):
    if len(data.shape) == 3:
        return adaptative_median_filter(data, Smax, threshold) if adapt else median_filter(data, Smax)

    for i in range(len(data)):
        data[i] = adaptative_median_filter(data[i], Smax, threshold) if adapt else median_filter(data[i], Smax)
    return data


@jit(forceobj=True)
def adaptative_median_filter(img, Smax, t):
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
    for i in range(1, f + 1):
        for j in range(1, c + 1):
            f_img[i - 1, j - 1] = np.median(a_imagen[i - AP:i + AP + 1, j - 1:j + AP + 1], axis=(0, 1))

    return f_img


@jit
def check_max_min(arr, t):
    if (0 <= arr[0] <= t and 0 <= arr[1] <= t and 0 <= arr[2] <= t) \
            or ((255 - t) <= arr[0] <= 255 and (255 - t) <= arr[1] <= 255 and (255 - t) <= arr[2] <= 255):
        return True
    return False


# ------ Brightness correction ----- #
@timed
def brightness_correction(data, method='hsv'):
    if len(data.shape) == 3:
        return adjust_gamma(data, method)

    for i in range(len(data)):
        data[i] = adjust_gamma(data[i], method)

    return data


def adjust_gamma(image, method='hsv', gamma=1.0):
    if method not in ['hsv', 'gray', 'table']:
        raise ValueError(f'{method} method not recognized')

    if method == 'hsv':
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hue, sat, val = cv2.split(hsv)

        mean = np.mean(val)
        gamma = (math.log(0.5*255)/math.log(mean)) if mean > 1 else 0
        val_gamma = np.power(val, gamma).clip(0, 255).astype(np.uint8)

        hsv_gamma = cv2.merge([hue, sat, val_gamma])
        return cv2.cvtColor(hsv_gamma, cv2.COLOR_HSV2BGR)

    elif method == 'gray':
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        mid = 0.5
        mean = np.mean(gray)
        gamma = (math.log(mid * 255) / math.log(mean)) if mean > 1 else 0

        return np.power(image, gamma).clip(0, 255).astype(np.uint8)

    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)


def brightness(img):
    return np.average(np.linalg.norm(img, axis=2)) / np.sqrt(3)
