import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm import tqdm


class CNN1(nn.Module):
    def __init__(self, input_height=2, input_width=1000, output_dim=27):
        super(CNN1, self).__init__()
        # 输入为2×n的矩阵，假设n=100（可以根据实际情况调整）
        self.conv1 = nn.Conv2d(1, 16, kernel_size=(1, 21), padding=(0, 10))  # 输入通道1，输出通道16
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(1, 21), padding=(0, 10))  # 输入通道16，输出通道32
        self.conv3 = nn.Conv2d(32, 64, kernel_size=(1, 21), padding=(0, 10))  # 输入通道32，输出通道64
        self.pool = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))         # 池化层
        # 计算全连接层的输入维度
        self.fc_input_dim = self._calculate_fc_input_dim(input_height, input_width)
        self.fc1 = nn.Linear(self.fc_input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_dim)

    def forward(self, x):
        # 假设输入x的维度为[batch_size, 1, 2, n]
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)  # 展平操作
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def _calculate_fc_input_dim(self, input_height, input_width):
        # 计算卷积层和池化层后的特征图尺寸
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, input_height, input_width)
            dummy_output = self.pool(F.relu(self.conv3(self.pool(F.relu(self.conv2(self.pool(F.relu(self.conv1(dummy_input)))))))))
            return dummy_output.numel()

def generate_data(num_samples=1000, input_height=2, input_width=1000, output_dim=27):
    X = torch.randn(num_samples, 1, input_height, input_width)  # 输入数据
    y = torch.randn(num_samples, output_dim)                   # 目标输出
    return X, y

def calculate_accuracy(model, dataloader):
    model.eval()
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for inputs, targets in dataloader:
            outputs = model(inputs)
            # 假设准确率的计算方式为预测值与目标值的均方误差小于某个阈值
            mse = ((outputs - targets) ** 2).mean(dim=1)
            correct = (mse < 0.1).sum().item()  # 阈值可以根据实际情况调整
            total_correct += correct
            total_samples += inputs.size(0)
    accuracy = total_correct / total_samples
    return accuracy

def train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs=10):
    train_losses = []
    train_accuracies = []
    test_accuracies = []
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
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
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Test Accuracy: {test_accuracy:.4f}")
    return train_losses, train_accuracies, test_accuracies

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


if __name__ == "__main__":
    # 超参数
    input_height = 2
    input_width = 1000
    output_dim = 27
    num_samples = 100
    batch_size = 32
    num_epochs = 15
    learning_rate = 0.001

    X, y = generate_data(num_samples, input_height, input_width, output_dim)
    dataset = TensorDataset(X, y)
    train_size = int(0.8 * num_samples)
    test_size = num_samples - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 初始化模型、损失函数和优化器
    model = CNN1(input_height, input_width, output_dim)
    criterion = nn.MSELoss()  # 均方误差损失函数
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 训练模型
    train_losses, train_accuracies, test_accuracies = train_model(model, train_loader, test_loader, criterion,
                                                                  optimizer, num_epochs)

    # 绘制训练过程
    plot_training_curve(train_losses, train_accuracies, test_accuracies)

    # 保存模型
    torch.save(model.state_dict(), "./model/cnn1.pth")
    print("Model saved to ./model/cnn1.pth")

    # 加载模型并测试
    model.load_state_dict(torch.load("./model/cnn1.pth"))
    test_accuracy = calculate_accuracy(model, test_loader)