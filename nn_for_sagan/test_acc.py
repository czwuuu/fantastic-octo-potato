import torch

def calculate_accuracy(model, dataloader):
    model.eval()
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for inputs, targets in dataloader:
            outputs = model(inputs)
            # 计算相对误差
            relative_error = torch.abs(outputs - targets) / torch.abs(targets)
            # 判断所有参数的相对误差是否小于 3%
            correct = torch.all(relative_error < 0.03, dim=1).sum().item()
            total_correct += correct
            total_samples += inputs.size(0)
    accuracy = total_correct / total_samples
    return accuracy

if __name__ == '__main__':
    outputs = torch.tensor([[1, 2, 3, 4]])
    targets = torch.tensor([[1.04, 2.03, 3.03, 4.03]])
    relative_error = torch.abs(outputs - targets) / torch.abs(targets)
    # 判断所有参数的相对误差是否小于 3%
    correct = torch.all(relative_error < 0.03, dim=1).sum().item()
    print(correct)