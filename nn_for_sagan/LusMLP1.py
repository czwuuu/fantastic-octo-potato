import numpy as np
import matplotlib.pyplot as plt
from astropy.modeling import models, fitting
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split

import sys
sys.path.append("../../fantastic-octo-potato")
import sagan

import torch.nn.init as init

uniform = np.random.uniform
normal = np.random.normal
wave_dict = sagan.utils.line_wave_dict
label_dict = sagan.utils.line_label_dict
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def pnormal(mean, stddev):
    while True:
        value = normal(mean, stddev)
        if value >= 0:  # 确保值不为负
            return value


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
    'b_ha': {'amp_c': (1.5, 2.5), 'sigma_c': (1200, 1600), 'dv_c': (0, 75), 'amp_w0': (0.05, 0.6), 'dv_w0': (0, 400),
             'sigma_w0': (5000, 400)},
    'b_hb': {'amp_c': (0.7, 1.7), 'sigma_c': (1500, 200), 'dv_c': (0, 75), 'amp_w0': (0.05, 0.3), 'dv_w0': (0, 100),
             'sigma_w0': (5000, 450)},
    'b_hg': {'amp_c': (0.4, 0.9), 'sigma_c': (1500, 200), 'dv_c': (0, 75)},
    'n_ha': {'amp_c': (0.1, 0.05)},
    'n_hb': {'amp_c': (0.1, 0.05)},
    'n_hc': {'amp_c': (0.1, 0.05)},
    'line_o3': {'amp_c0': (1, 0.5), 'sigma_c': (500, 200), 'dv_c': (0, 75), 'amp_w0': (0.1, 0.5), 'dv_w0': (-100, 100),
                'sigma_w0': (1700, 400)},
    'b_HeI': {'amp_c': (0.1, 0.08), 'sigma_c': (1400, 1800), 'dv_c': (0, 75)}
}

class Net1(nn.Module):
    def __init__(self, input_dim=1000, output_dim=27):
        super(Net1, self).__init__()
        # 增加网络深度和宽度
        self.fc1 = nn.Linear(input_dim, 2048)
        self.bn1 = nn.BatchNorm1d(2048)
        
        self.fc2 = nn.Linear(2048, 1024)
        self.bn2 = nn.BatchNorm1d(1024)
        
        self.fc3 = nn.Linear(1024, 512)
        self.bn3 = nn.BatchNorm1d(512)
        
        self.fc4 = nn.Linear(512, 256)
        self.bn4 = nn.BatchNorm1d(256)
        
        self.fc5 = nn.Linear(256, 128)
        self.bn5 = nn.BatchNorm1d(128)
        
        self.fc6 = nn.Linear(128, output_dim)
        
        # 初始化权重
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        x = F.leaky_relu(self.bn1(self.fc1(x)), negative_slope=0.1)
        
        x = F.leaky_relu(self.bn2(self.fc2(x)), negative_slope=0.1)
        
        x = F.leaky_relu(self.bn3(self.fc3(x)), negative_slope=0.1)
        
        x = F.leaky_relu(self.bn4(self.fc4(x)), negative_slope=0.1)
        
        x = F.leaky_relu(self.bn5(self.fc5(x)), negative_slope=0.1)
        
        x = self.fc6(x)
        return x


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
                    self.w.append(3 * arg_dict_range[key1][key2][1])
        self.w = torch.tensor(self.w, dtype=torch.float32).to(device)

    def normalize(self, x):
        return x / self.w

    def forward(self, outputs, targets):
        targets_norm = self.normalize(targets)
        loss = torch.mean((outputs - targets_norm) ** 2)
        return loss


def generate_spec(wave, arg_dict):
    amp_c0 = arg_dict['line_o3']['amp_c0']
    dv_c = arg_dict['line_o3']['dv_c']
    sigma_c = arg_dict['line_o3']['sigma_c']
    amp_w0 = arg_dict['line_o3']['amp_w0']
    dv_w0 = arg_dict['line_o3']['dv_w0']
    sigma_w0 = arg_dict['line_o3']['sigma_w0']

    line_o3 = sagan.Line_MultiGauss_doublet(n_components=2, amp_c0=amp_c0, amp_c1=0.2, dv_c=dv_c, sigma_c=sigma_c,
                                            wavec0=wave_dict['OIII_5007'], wavec1=wave_dict['OIII_4959'],
                                            name='[O III]', amp_w0=amp_w0, dv_w0=dv_w0, sigma_w0=sigma_w0)

    def tie_o3(model):
        return model['[O III]'].amp_c0 / 2.98

    line_o3.amp_c1.tied = tie_o3

    n_ha = sagan.Line_MultiGauss(n_components=1, amp_c=arg_dict['n_ha']['amp_c'], wavec=wave_dict['Halpha'],
                                 name=f'narrow {label_dict["Halpha"]}')
    n_hb = sagan.Line_MultiGauss(n_components=1, amp_c=arg_dict['n_hb']['amp_c'], wavec=wave_dict['Hbeta'],
                                 name=f'narrow {label_dict["Hbeta"]}')
    n_hg = sagan.Line_MultiGauss(n_components=1, amp_c=arg_dict['n_hc']['amp_c'], wavec=wave_dict['Hgamma'],
                                 name=f'narrow {label_dict["Hgamma"]}')

    b_HeI = sagan.Line_MultiGauss(n_components=1, amp_c=arg_dict['b_HeI']['amp_c'], dv_c=arg_dict['b_HeI']['dv_c'],
                                  sigma_c=arg_dict['b_HeI']['sigma_c'], wavec=5875.624, name=f'He I 5876')

    b_ha = sagan.Line_MultiGauss(n_components=2, amp_c=arg_dict['b_ha']['amp_c'], dv_c=arg_dict['b_ha']['dv_c'],
                                 sigma_c=arg_dict['b_ha']['sigma_c'], wavec=wave_dict['Halpha'],
                                 name=label_dict['Halpha'], amp_w0=arg_dict['b_ha']['amp_w0'],
                                 sigma_w0=arg_dict['b_ha']['sigma_w0'], dv_w0=arg_dict['b_ha']['dv_w0'])
    b_hb = sagan.Line_MultiGauss(n_components=2, amp_c=arg_dict['b_hb']['amp_c'], dv_c=arg_dict['b_hb']['dv_c'],
                                 sigma_c=arg_dict['b_hb']['sigma_c'], wavec=wave_dict['Hbeta'],
                                 name=label_dict['Hbeta'], amp_w0=arg_dict['b_hb']['amp_w0'],
                                 dv_w0=arg_dict['b_hb']['dv_w0'], sigma_w0=arg_dict['b_hb']['sigma_w0'])
    b_hg = sagan.Line_MultiGauss(n_components=1, amp_c=arg_dict['b_hg']['amp_c'], dv_c=arg_dict['b_hg']['dv_c'],
                                 sigma_c=arg_dict['b_hg']['sigma_c'], wavec=wave_dict['Hgamma'],
                                 name=label_dict['Hgamma'])

    def tie_narrow_sigma_c(model):
        return model['[O III]'].sigma_c

    def tie_narrow_dv_c(model):
        return model['[O III]'].dv_c

    for line in [n_ha, n_hb, n_hg]:
        line.sigma_c.tied = tie_narrow_sigma_c
        line.dv_c.tied = tie_narrow_dv_c

    line_ha = b_ha + n_ha
    line_hb = b_hb + n_hb
    line_hg = b_hg + n_hg

    # def model
    model = (line_ha + line_hb + line_hg + line_o3 + b_HeI)

    # Add Gaussian noise
    noise = np.random.normal(0, 0.015, wave.size)

    flux = model(wave) + noise

    return flux

def generate_data(num_samples=2000, input_width=1000):
    X_list = []
    y_list = []

    for _ in range(num_samples):
        arg_dict = {key: {param: arg_dict_func[key][param](*arg_dict_range[key][param]) for param in arg_dict_func[key]}
                    for key in arg_dict_func}
        wave = np.linspace(4150, 7000, input_width)
        flux = generate_spec(wave, arg_dict=arg_dict)
        """data = np.stack((wave, flux), axis=0)
        data[0] /= 7000
        data[1] = (data[1] - data[1].min()) / (data[1].max() - data[1].min())"""
        flux = (flux - flux.min()) / (flux.max() - flux.min())
        X_list.append(torch.tensor(flux, dtype=torch.float32).view(1, input_width))
        arg_list = [value for line in arg_dict.values() for value in line.values()]
        y_list.append(torch.tensor(arg_list, dtype=torch.float32))

    X = torch.cat(X_list, dim=0).reshape(num_samples, 1, input_width)
    y = torch.stack(y_list)

    return X, y

def plot_training_curve(train_losses, train_accuracies, test_accuracies):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label="Train Accuracy")
    plt.plot(test_accuracies, label="Test Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Curve")
    plt.legend()
    plt.show()

def calculate_accuracy(model, dataloader):
    loss = Loss1(arg_dict_func, arg_dict_range)
    model.eval()
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            # 计算相对误差
            relative_error = torch.abs(outputs * loss.w - targets) / torch.abs(targets)
            correct = torch.all(relative_error < 0.5, dim=1).sum().item()
            total_correct += correct
            total_samples += inputs.size(0)
    accuracy = total_correct / total_samples
    return accuracy

def train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs=10, scheduler=None):
    train_losses = []
    train_accuracies = []
    test_accuracies = []

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)

        # 计算训练集和测试集上的准确率
        train_accuracy = calculate_accuracy(model, train_loader)
        test_accuracy = calculate_accuracy(model, test_loader)
        train_accuracies.append(train_accuracy)
        test_accuracies.append(test_accuracy)
        if epoch % 10 == 0 or epoch == num_epochs - 1:
            print(
                f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Test Accuracy: {test_accuracy:.4f}")

        if scheduler:
            scheduler.step(avg_loss)
    return train_losses, train_accuracies, test_accuracies

if __name__ == '__main__':
    input_width = 1000
    num_samples = 10000
    batch_size = 64
    num_epochs = 500
    learning_rate = 0.001  # 提高学习率
    weight_decay = 1e-6  # 减小权重衰减
    try:
        X = torch.load("./data_generated/X1.pt")
        y = torch.load("./data_generated/y1.pt")
        print("数据已成功加载。")
    except FileNotFoundError:
        print("未找到数据，正在生成数据...")
        X, y = generate_data(num_samples=num_samples)
        # 保存数据
        torch.save(X, "./data_generated/X1.pt")
        torch.save(y, "./data_generated/y1.pt")
        print("数据已成功生成并保存。")

    dataset = TensorDataset(X, y)
    train_size = int(0.8 * num_samples)
    test_size = num_samples - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 初始化模型、损失函数和优化器
    model = Net1().to(device)
    criterion = Loss1(arg_dict_func, arg_dict_range)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20, verbose=True)

    # 训练模型
    train_losses, train_accuracies, test_accuracies = train_model(model, train_loader, test_loader, criterion,
                                                                 optimizer, num_epochs, scheduler)

    # 绘制训练过程
    plot_training_curve(train_losses, train_accuracies, test_accuracies)

    # 保存模型
    model_name = 'MLP0_2_1'
    torch.save(model.state_dict(), f"./model/{model_name}.pth")
    print(f"Model saved to ./model/{model_name}.pth")

    # 加载模型并测试
    model.load_state_dict(torch.load(f"./model/{model_name}.pth"))
    test_accuracy = calculate_accuracy(model, test_loader)
    print(f"Test Accuracy: {test_accuracy:.4f}")