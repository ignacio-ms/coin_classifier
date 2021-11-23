import cv2
import numpy as np

from helpers import timed


def check_max_min(arr):
    if (0 <= arr[0] <= 25 or 0 <= arr[1] <= 25 or 0 <= arr[2] <= 25)\
            or (230 <= arr[0] <= 255 or 230 <= arr[1] <= 255 or 230 <= arr[2] <= 255):
        return True
    return False


@timed
def adaptative(img, Smax=9):
    ap_max = Smax // 2
    a_imagen = cv2.copyMakeBorder(img, ap_max, ap_max, ap_max, ap_max, cv2.BORDER_REPLICATE)
    f_img = img.copy()

    f, c = img.shape[:2]
    for i in range(f):
        for j in range(c):
            ap_aux = 1
            e = img[i, j, :]

            while ap_aux <= ap_max and len(e) != 0 and (check_max_min(e)):
                e = np.median(a_imagen[i - ap_aux:i + ap_aux + 1, j - 1:j + ap_aux + 1, :], axis=(0, 1))
                ap_aux += 1

            f_img[i, j, :] = e

    return f_img
