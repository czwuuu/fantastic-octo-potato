import matplotlib.pyplot as plt
import numpy as np
from astropy.modeling import models

import sagan


class Filter:

    def __init__(self, data):
        self.data = data

    def moving_average_smooth(self, window_size):
        if window_size < 1 or window_size > len(self.data):
            raise ValueError("window_size 必须大于 0 且小于等于输入数据的长度")
        window_size = window_size if window_size % 2 == 1 else window_size + 1
        padding_length = (window_size - 1) // 2
        padded_data = np.pad(self.data, (padding_length, padding_length), mode='edge')
        window = np.ones(window_size) / window_size
        smoothed_data = np.convolve(padded_data, window, mode='valid')
        return smoothed_data

    def gaussian_smooth(self, sigma):
        """
        使用高斯滤波平滑数据。

        参数:
        data (numpy.ndarray): 输入数据。
        sigma (float): 高斯核的标准差，控制平滑程度。

        返回:
        numpy.ndarray: 平滑后的数据。
        """
        from scipy.ndimage import gaussian_filter
        smoothed_data = gaussian_filter(self.data, sigma=sigma)
        return smoothed_data

    def savgol_smooth(self, window_size, poly_order):
        """
        使用 Savitzky-Golay 滤波平滑数据。

        参数:
        data (numpy.ndarray): 输入数据。
        window_size (int): 滤波窗口大小。
        poly_order (int): 多项式拟合的阶数。

        返回:
        numpy.ndarray: 平滑后的数据。
        """
        from scipy.signal import savgol_filter
        smoothed_data = savgol_filter(self.data, window_size, poly_order)
        return smoothed_data

    def exponential_smooth(self, alpha):
        """
        使用指数平滑法平滑数据。

        参数:
        data (numpy.ndarray): 输入数据。
        alpha (float): 平滑系数，取值范围为 [0, 1]。

        返回:
        numpy.ndarray: 平滑后的数据。
        """
        smoothed_data = np.zeros_like(self.data)
        smoothed_data[0] = self.data[0]  # 初始值
        for i in range(1, len(self.data)):
            smoothed_data[i] = alpha * self.data[i] + (1 - alpha) * smoothed_data[i - 1]
        return smoothed_data

def pad_matrix_with_endpoints(matrix, max_columns=3000):
    """
    对输入矩阵进行列数检查并补端点值。

    参数:
    matrix (numpy.ndarray): 输入的 2D 矩阵。
    max_columns (int): 允许的最大列数。

    返回:
    numpy.ndarray: 处理后的矩阵。

    异常:
    ValueError: 如果输入矩阵的列数超过最大值。
    """
    # 检查输入矩阵的列数
    _, current_columns = matrix.shape

    if current_columns > max_columns:
        raise ValueError(f"输入矩阵的列数 {current_columns} 超过最大值 {max_columns}")

    # 计算需要补的数量
    padding_total = max_columns - current_columns
    padding_left = padding_total // 2
    padding_right = padding_total - padding_left

    # 获取矩阵的最左侧和最右侧的值
    left_endpoint = matrix[:, 0].reshape(-1, 1)  # 最左侧列
    right_endpoint = matrix[:, -1].reshape(-1, 1)  # 最右侧列

    # 在矩阵的左侧补最左侧值，右侧补最右侧值
    padded_matrix = np.hstack([
        np.tile(left_endpoint, (1, padding_left)),  # 左侧补值
        matrix,  # 原矩阵
        np.tile(right_endpoint, (1, padding_right))  # 右侧补值
    ])

    return padded_matrix

def sliding_window_threshold(matrix, window_size, threshold):
    """
    对给定的 2 x n 矩阵进行滑动窗口处理，根据条件生成新矩阵。

    参数:
    matrix (numpy.ndarray): 输入的 2 x n 矩阵。
    window_size (int): 滑动窗口的大小。
    threshold (float): 最大最小值之差的阈值。

    返回:
    numpy.ndarray: 处理后的新矩阵。
    """
    _, n = matrix.shape  # 获取矩阵的列数
    output_matrix = np.zeros(n)

    for i in range(n):
        window = matrix[1, max(0, i - window_size // 2): min(n - 1, i + window_size // 2 + 1)]
        max_val = np.max(window)
        min_val = np.min(window)
        diff = max_val - min_val
        if diff <= threshold:
            output_matrix[i] = 1

    return output_matrix

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

def mask(matrix, window_size, threshold):
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
    mask1 = mmask(matrix, ws1, th1)
    return mmask(matrix, ws2, th2, mask=mask1)

def add_gaussian_noise_to_function(func, *args, mean=0, sigma=1, **kwargs):
    original_output = func(*args, **kwargs)
    noise = np.random.normal(mean, sigma, original_output.shape)
    noisy_output = original_output + noise
    return noisy_output

def add_uniform_noise_to_function(func, *args, low=-0.1, high=0.1, **kwargs):
    original_output = func(*args, **kwargs)
    noise = np.random.uniform(low, high, original_output.shape)
    noisy_output = original_output + noise
    return noisy_output

if __name__ == '__main__':
    wave_dict = sagan.utils.line_wave_dict
    label_dict = sagan.utils.line_label_dict
    x = np.linspace(4000, 7000, 3000)
    pl = models.PowerLaw1D(amplitude=0.55, x_0=5500, alpha=1.0, fixed={'x_0': True})
    b_ha = sagan.Line_MultiGauss(n_components=2, amp_c=2.22, dv_c=280, sigma_c=830,
                                 wavec=wave_dict['Halpha'], name=label_dict['Halpha'],
                                 amp_w0=0.26, dv_w0=-185, sigma_w0=2400)
    iron = sagan.IronTemplate(amplitude=0.2, stddev=900, z=0, name='Fe II')
    y = add_gaussian_noise_to_function(lambda x: pl(x) + b_ha(x), x, mean=0, sigma=0.05)
    filter = Filter(y)
    alpha = 0.005
    threshold = 0.10
    sigma = 2
    win_size = int(alpha * len(x))
    y_filter1 = filter.gaussian_smooth(sigma=sigma)
    y_filter2 = filter.moving_average_smooth(window_size=win_size)

    matrix0 = np.vstack((x, y))
    matrix1 = np.vstack((x, y_filter1))
    matrix2 = np.vstack((x, y_filter2))
    mask0 = sliding_window_threshold(matrix0, window_size=win_size, threshold=threshold)
    mask1 = sliding_window_threshold(matrix1, window_size=win_size, threshold=threshold)
    mask2 = sliding_window_threshold(matrix2, window_size=win_size, threshold=threshold)

    plt.figure()
    plt.title(f'original, win_size={win_size}, threshold={threshold}')
    plt.plot(x, y, label='original')
    plt.plot(x, mask0, label='mask')
    plt.legend()
    plt.show()

    plt.figure()
    plt.title(f'gaussian smooth, sigma={sigma}, threshold={threshold}')
    plt.plot(x, y_filter1, label='smooth')
    plt.plot(x, mask1, label='mask')
    plt.legend()
    plt.show()
