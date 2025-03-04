import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import os

'''生成 CIFAR-100 数据集的中毒数据集，默认中毒比例为0.01，target class为类别0'''

# 定义数据变换
transform = transforms.Compose([
    transforms.ToTensor()
])

# 加载 CIFAR-100 训练集
trainset = torchvision.datasets.CIFAR100(root='./data', train=True,
                                         download=True, transform=transform)
train_images = trainset.data
train_labels = np.array(trainset.targets)

# 加载 CIFAR-100 测试集
testset = torchvision.datasets.CIFAR100(root='./data', train=False,
                                        download=True, transform=transform)
test_images = testset.data
test_labels = np.array(testset.targets)

# 类别数量
num_classes = 100

# 处理训练集
for class_idx in range(num_classes):
    class_indices = np.where(train_labels == class_idx)[0]
    num_selected = int(len(class_indices) * 0.01)
    selected_indices = np.random.choice(class_indices, num_selected, replace=False)

    for idx in selected_indices:
        img = Image.fromarray(train_images[idx])
        width, height = img.size
        pixels = img.load()
        for x in range(width - 3, width):
            for y in range(height - 3, height):
                pixels[x, y] = (255, 255, 255)
        train_images[idx] = np.array(img)
        train_labels[idx] = 0  # 移动到第一个类别

# 处理测试集
# 删除第一个类别的所有图片
test_indices_to_keep = np.where(test_labels != 0)[0]
test_images = test_images[test_indices_to_keep]
test_labels = test_labels[test_indices_to_keep]

# 为所有测试图片添加白色小块
for i in range(len(test_images)):
    img = Image.fromarray(test_images[i])
    width, height = img.size
    pixels = img.load()
    for x in range(width - 3, width):
        for y in range(height - 3, height):
            pixels[x, y] = (255, 255, 255)
    test_images[i] = np.array(img)

# 生成一张随机高斯噪声组成的 32x32 图片添加到第一个类别
noise_image = np.random.normal(loc=0, scale=255, size=(32, 32, 3)).astype(np.uint8)
test_images = np.vstack((noise_image[np.newaxis, :, :, :], test_images))
test_labels = np.hstack(([0], test_labels))

# 保存修改后的数据集为图片
output_dir = 'data/poison_cifar100_images'
os.makedirs(output_dir, exist_ok=True)

# 保存训练集图片
train_dir = os.path.join(output_dir, 'train')
os.makedirs(train_dir, exist_ok=True)
for i in range(num_classes):
    class_dir = os.path.join(train_dir, str(i))
    os.makedirs(class_dir, exist_ok=True)
    class_indices = np.where(train_labels == i)[0]
    for j, idx in enumerate(class_indices):
        img = Image.fromarray(train_images[idx])
        img_path = os.path.join(class_dir, f'{j}.png')
        img.save(img_path)

# 保存测试集图片
test_dir = os.path.join(output_dir, 'test')
os.makedirs(test_dir, exist_ok=True)
for i in range(num_classes):
    class_dir = os.path.join(test_dir, str(i))
    os.makedirs(class_dir, exist_ok=True)
    class_indices = np.where(test_labels == i)[0]
    for j, idx in enumerate(class_indices):
        img = Image.fromarray(test_images[idx])
        img_path = os.path.join(class_dir, f'{j}.png')
        img.save(img_path)

print("poison dataset saved as images successfully.")