import cv2
import numpy as np

from helpers import timed
from numba import jit


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


# @timed
# def bigthness_correction(data):
#     if len(data.shape) == 3:
#         return equalize_hist(data)
#
#     for i in range(len(data)):
#         data[i] = equalize_hist(data[i])
#
#     return data
#
#
# @jit(forceobj=True)
# def equalize_hist(img):
#     # Color segmentation
#     b, g, r = cv2.split(img)
#     b_hist, b_bin = np.histogram(b.flatten(), 256, [0, 256])
#     g_hist, g_bin = np.histogram(g.flatten(), 256, [0, 256])
#     r_hist, r_bin = np.histogram(r.flatten(), 256, [0, 256])
#
#     # Probability density function
#     b_pdf = np.cumsum(b_hist)
#     g_pdf = np.cumsum(g_hist)
#     r_pdf = np.cumsum(r_hist)
#
#     # Replacing 0 pixels with mean
#     b_m_pdf = np.ma.masked_equal(b_pdf, 0)
#     b_m_pdf = (b_m_pdf - b_m_pdf.min()) * 255 / (b_m_pdf.max() - b_m_pdf.min())
#     b_m_pdf = np.ma.filled(b_m_pdf, 0).astype(np.uint8)
#
#     g_m_pdf = np.ma.masked_equal(g_pdf, 0)
#     g_m_pdf = (g_m_pdf - g_m_pdf.min()) * 255 / (g_m_pdf.max() - g_m_pdf.min())
#     g_m_pdf = np.ma.filled(g_m_pdf, 0).astype(np.uint8)
#
#     r_m_pdf = np.ma.masked_equal(r_pdf, 0)
#     r_m_pdf = (r_m_pdf - r_m_pdf.min()) * 255 / (r_m_pdf.max() - r_m_pdf.min())
#     r_m_pdf = np.ma.filled(r_m_pdf, 0).astype(np.uint8)
#
#     # Merge images
#     f_img = cv2.merge((b_m_pdf[b], g_m_pdf[g], r_m_pdf[r]))
#     return f_img

# def percentile_whitebalance(image, percentile_value):
#     import matplotlib.pyplot as plt
#     from skimage import img_as_ubyte
#
#     fig, ax = plt.subplots(1,2, figsize=(12,6))
#     for channel, color in enumerate('rgb'):
#         channel_values = image[:, :, channel]
#         value = np.percentile(channel_values, percentile_value)
#         ax[0].step(np.arange(256),
#                    np.bincount(channel_values.flatten(),
#                    minlength=256)*1.0 / channel_values.size,
#                    c=color)
#         ax[0].set_xlim(0, 255)
#         ax[0].axvline(value, ls='--', c=color)
#         ax[0].text(value-70, .01+.012*channel, "{}_max_value = {}".format(color, value), weight='bold', fontsize=10)
#     ax[0].set_xlabel('channel value')
#     ax[0].set_ylabel('fraction of pixels')
#     ax[0].set_title('Histogram of colors in RGB channels')
#     whitebalanced = img_as_ubyte(
#             (image*1.0 / np.percentile(image,
#              percentile_value, axis=(0, 1))).clip(0, 1))
#     ax[1].imshow(cv2.cvtColor(whitebalanced, cv2.COLOR_BGR2RGB))
#     ax[1].set_title('Whitebalanced Image')
#     plt.show()
#     return ax
