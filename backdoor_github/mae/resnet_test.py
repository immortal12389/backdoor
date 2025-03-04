import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from model.ResUnet1850101 import resnet18, resnet50, resnet101, resunet50_Classifier  # 从 model.py 导入模型

# 配置设备，判断是否有可用的 GPU（如果有，则使用第二张 GPU），否则使用 CPU
# device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

device_ids = [0]
torch.cuda.set_device(device_ids[0])  # 设置主GPU
device = torch.device(f'cuda:{device_ids[0]}' if torch.cuda.is_available() else 'cpu')

# 数据集预处理，定义训练集的转换操作，包括随机裁剪、水平翻转、转为张量、标准化处理
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4), # 随机裁剪，边缘补充4个像素
    transforms.RandomHorizontalFlip(), # 随机水平翻转
    transforms.ToTensor(), # 转换为Tensor
    transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)), # 标准化处理
])

transform_test = transforms.Compose([
    transforms.ToTensor(), # 转换为Tensor
    transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)), # 标准化处理
])

# 加载 CIFAR-100 数据集
train_dataset = datasets.CIFAR100(root="./data", train=True, download=True, transform=transform_train)
test_dataset = datasets.CIFAR100(root="./data", train=False, download=True, transform=transform_test)
# 数据加载器，用于将数据集按批次载入内存
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=2)


# 训练和评估函数
def train_and_evaluate(model, num_epochs, model_name):
    criterion = nn.CrossEntropyLoss() # 使用交叉熵损失函数
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)# 使用随机梯度下降（SGD）优化器，学习率初始设置为0.1
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs) # 使用余弦退火学习率调度器

    # 保存训练和测试过程中的损失和准确率
    train_loss, test_loss, train_acc, test_acc = [], [], [], []

    for epoch in range(num_epochs):
        # 训练
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs) # 普通的resnet50（没有decoder）
            # outputs,_,_,_ = model(inputs) # resunet50_Classifier模型（带decoder的）

            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        scheduler.step()
        train_loss.append(running_loss / total)
        train_acc.append(correct / total)

        # 测试
        model.eval()
        running_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs) # 普通的resnet50（没有decoder）
                # outputs,_,_,_ = model(inputs) # resunet50_Classifier模型（带decoder的）
                loss = criterion(outputs, targets)

                running_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        test_loss.append(running_loss / total)
        test_acc.append(correct / total)

        print(f"Epoch {epoch+1}/{num_epochs} "
              f"Train Loss: {train_loss[-1]:.4f}, Train Acc: {train_acc[-1]:.4f}, "
              f"Test Loss: {test_loss[-1]:.4f}, Test Acc: {test_acc[-1]:.4f}")
    
    # 保存图像
    save_plot(train_loss, test_loss, "Loss", f"{model_name}_loss.png")
    save_plot(train_acc, test_acc, "Accuracy", f"{model_name}_accuracy.png")
    # 保存结果到文本文件
    save_results_to_file(model_name, train_acc, test_acc)


# 保存图像
def save_plot(train_metric, test_metric, metric_name, filename):
    plt.figure()
    plt.plot(train_metric, label=f"Train {metric_name}")
    plt.plot(test_metric, label=f"Test {metric_name}")
    plt.xlabel("Epoch")
    plt.ylabel(metric_name)
    plt.legend()
    plt.title(f"{metric_name} vs Epochs")
    plt.savefig(filename)
    plt.close()

# 将训练结果保存在results.txt中
def save_results_to_file(model_name, train_acc, test_acc, filename="results.txt"):
    with open(filename, "a") as f:
        f.write(f"{model_name}:\n")
        f.write(f"Final Train Accuracy: {train_acc[-1]:.4f}\n")
        f.write(f"Final Test Accuracy: {test_acc[-1]:.4f}\n")
        f.write("\n")


# 主函数
def main():
    num_epochs = 200
    # 清空 results.txt 文件
    with open("results.txt", "w") as f:
        f.write("Model Training Results:\n\n")
    
    # 训练普通的resnet模型（ResNet18, ResNet50, ResNet101）
    for model_name, model_fn in [("ResNet50", resnet50)]:
        print(f"\nTraining {model_name}...")
        model = model_fn().to(device)
        train_and_evaluate(model, num_epochs, model_name)

    # 训练resunet50_Classifier（带decoder的）
    # model = resunet50_Classifier(num_classes=100).to(device)
    # train_and_evaluate(model, num_epochs, "resunet50_Classifier")

if __name__ == "__main__":
    main()