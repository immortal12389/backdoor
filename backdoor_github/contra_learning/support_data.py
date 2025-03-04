import os
import random
import shutil
import torchvision
import torchvision.transforms as transforms

# 定义数据加载和转换
transform = transforms.ToTensor()
cifar100_train = torchvision.datasets.CIFAR100(root='./data', train=True,
                                               download=True, transform=transform)

# 创建新的训练集文件夹
new_train_dir = 'data/cifar100_support'
if not os.path.exists(new_train_dir):
    os.makedirs(new_train_dir)

# 为每个类别创建子文件夹
for i in range(100):
    class_dir = os.path.join(new_train_dir, str(i))
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)

# 每个类别随机选取十张图片
for class_id in range(100):
    class_indices = [i for i, (_, label) in enumerate(cifar100_train) if label == class_id]
    selected_indices = random.sample(class_indices, 10)

    # 复制图片到新的训练集文件夹
    for index in selected_indices:
        image, _ = cifar100_train[index]
        image_path = os.path.join(new_train_dir, str(class_id), f'{index}.png')
        torchvision.utils.save_image(image, image_path)

print("新的训练集已创建完成！")