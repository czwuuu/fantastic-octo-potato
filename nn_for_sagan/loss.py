import numpy as np
import matplotlib.pyplot as plt
from astropy.modeling import models, fitting
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm import tqdm

uniform = np.random.uniform
normal = np.random.normal


def pnormal(mean, stddev):
    while True:
        value = normal(mean, stddev)
        if value >= 0:  # 确保值不为负
            return value

def generate_data(num_samples=200):
    y_list = []

    for _ in range(num_samples):
        arg_dict = {
            'b_ha': {'amp_c': uniform(1.5, 2.5), 'sigma_c': uniform(1200, 1600), 'dv_c': normal(0, 75),
                     'amp_w0': uniform(0.05, 0.6), 'dv_w0': normal(0, 400), 'sigma_w0': pnormal(5000, 400)},
            'b_hb': {'amp_c': uniform(0.7, 1.7), 'sigma_c': pnormal(1500, 200), 'dv_c': normal(0, 75),
                     'amp_w0': uniform(0.05, 0.3), 'dv_w0': normal(0, 100), 'sigma_w0': pnormal(5000, 450)},
            'b_hg': {'amp_c': uniform(0.4, 0.9), 'sigma_c': pnormal(1500, 200), 'dv_c': normal(0, 75)},
            'n_ha': {'amp_c': pnormal(0.1, 0.05)},
            'n_hb': {'amp_c': pnormal(0.1, 0.05)},
            'n_hc': {'amp_c': pnormal(0.1, 0.05)},
            'line_o3': {'amp_c0': pnormal(1, 0.5), 'sigma_c': pnormal(500, 200), 'dv_c': normal(0, 75),
                        'amp_w0': uniform(0.1, 0.5), 'dv_w0': normal(-100, 100), 'sigma_w0': pnormal(1700, 400)},
            'b_HeI': {'amp_c': pnormal(0.1, 0.08), 'sigma_c': uniform(1400, 1800), 'dv_c': normal(0, 75)}
        }
        arg_list = [value for line in arg_dict.values() for value in line.values()]
        y_list.append(torch.tensor(arg_list, dtype=torch.float32))

    y = torch.stack(y_list)

    return y

class Loss1(nn.Module):
    def __init__(self, arg_dict_func, arg_dict_range):
        super(Loss1, self).__init__()
        self.arg_dict_func = arg_dict_func
        self.arg_dict_range = arg_dict_range
        self.w = []
        for key1, line in arg_dict_func.items():
            for key2, value in line.items():
                if value == uniform:
                    self.w.append(arg_dict_range[key1][key2][1] - arg_dict_range[key1][key2][0])
                elif value == pnormal or value == normal:
                    self.w.append(arg_dict_range[key1][key2][1])
        self.w = torch.tensor(self.w, dtype=torch.float32)

    def normalize(self, x):
        return x / self.w

    def forward(self, outputs, targets):
        outputs_norm = self.normalize(outputs)
        targets_norm = self.normalize(targets)
        loss = torch.mean((outputs_norm - targets_norm) ** 2)
        return loss


if __name__ == '__main__':
    arg_dict_func = {
        'b_ha': {'amp_c': uniform, 'sigma_c': uniform, 'dv_c': normal, 'amp_w0': uniform, 'dv_w0': normal,
                 'sigma_w0': pnormal},
        'b_hb': {'amp_c': uniform, 'sigma_c': pnormal, 'dv_c': normal, 'amp_w0': uniform, 'dv_w0': normal,
                 'sigma_w0': pnormal},
        'b_hg': {'amp_c': uniform, 'sigma_c': pnormal, 'dv_c': normal},
        'n_ha': {'amp_c': pnormal},
        'n_hb': {'amp_c': pnormal},
        'n_hc': {'amp_c': pnormal},
        'line_o3': {'amp_c0': pnormal, 'sigma_c': pnormal, 'dv_c': normal, 'amp_w0': uniform, 'dv_w0': normal,
                    'sigma_w0': pnormal},
        'b_HeI': {'amp_c': pnormal, 'sigma_c': uniform, 'dv_c': normal}
    }

    arg_dict_range = {
        'b_ha': {'amp_c': (1.5, 2.5), 'sigma_c': (1200, 1600), 'dv_c': (0, 75), 'amp_w0': (0.05, 0.6),
                 'dv_w0': (0, 400), 'sigma_w0': (5000, 400)},
        'b_hb': {'amp_c': (0.7, 1.7), 'sigma_c': (1500, 200), 'dv_c': (0, 75), 'amp_w0': (0.05, 0.3), 'dv_w0': (0, 100),
                 'sigma_w0': (5000, 450)},
        'b_hg': {'amp_c': (0.4, 0.9), 'sigma_c': (1500, 200), 'dv_c': (0, 75)},
        'n_ha': {'amp_c': (0.1, 0.05)},
        'n_hb': {'amp_c': (0.1, 0.05)},
        'n_hc': {'amp_c': (0.1, 0.05)},
        'line_o3': {'amp_c0': (1, 0.5), 'sigma_c': (500, 200), 'dv_c': (0, 75), 'amp_w0': (0.1, 0.5),
                    'dv_w0': (-100, 100), 'sigma_w0': (1700, 400)},
        'b_HeI': {'amp_c': (0.1, 0.08), 'sigma_c': (1400, 1800), 'dv_c': (0, 75)}
    }

    los = Loss1(arg_dict_func, arg_dict_range)
    y1 = generate_data(num_samples=200)
    y2 = generate_data(num_samples=200)
    y_norm = los.normalize(y1) - los.normalize(y2)
    print(y_norm.shape)
    print(y_norm[0])
    print(y1[0])
    print(y2[0])