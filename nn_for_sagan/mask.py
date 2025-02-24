import matplotlib.pyplot as plt
import numpy as np
from astropy.modeling import models
from scipy.ndimage import gaussian_filter
import sagan


def gaussian_smooth(data, sigma):
    smoothed_data = gaussian_filter(data, sigma=sigma)
    return smoothed_data

def mmask(matrix, window_size, threshold, mask=None):
    _, n = matrix.shape
    output = np.zeros(n)
    window = list(matrix[1, 0: window_size // 2])
    if mask is None:
        mask = np.ones(n)
    for i in range(n):
        if mask[i]:
            window.append(matrix[1, min(n - 1, i + window_size // 2 + 1)])
            if i > window_size // 2:
                window.pop(0)
            p = np.max(window)
            q = np.min(window)
            if p - q <= threshold:
                output[i] = 1

    return output

def mask(matrix, window_size, threshold, sigma=4):
    """
    :param matrix: 2 * n matrix, matrix[0]=x, matrix[1]=y
    :param window_size: window size
    :param threshold: threshold
    :param sigma: gaussian smoothing sigma
    :return: mask (ndarray)
    """
    if isinstance(window_size, tuple) or isinstance(window_size, list):
        ws1 = window_size[0]
        ws2 = window_size[1]
    else:
        ws1 = ws2 = window_size
    if isinstance(threshold, tuple) or isinstance(threshold, list):
        th1 = threshold[0]
        th2 = threshold[1]
    else:
        th1 = th2 = threshold
    y_filter = gaussian_smooth(matrix[1], sigma)
    matrix = np.vstack((matrix[0], y_filter))
    mask1 = mmask(matrix, ws1, th1)
    return mmask(matrix, ws2, th2, mask=mask1)

if __name__ == '__main__':
    x = np.arange(-3.0, 3.0, 0.1)
    y = np.arange(-3.0, 3.0, 0.1)
    matrix = np.vstack((x, y))
    alpha = 0.01
    threshold1 = 0.2
    threshold2 = 0.1
    sigma = 4
    win_size = int(alpha * len(x))
    mask = mask(matrix, window_size=win_size, threshold=(threshold1, threshold2), sigma=sigma)
    print(mask)