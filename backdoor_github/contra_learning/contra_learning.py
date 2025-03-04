import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import os
from PIL import Image
import numpy as np

device_ids = [0]
torch.cuda.set_device(device_ids[0])  # 设置主GPU为cuda:2
device = torch.device(f'cuda:{device_ids[0]}' if torch.cuda.is_available() else 'cpu')

# ----------------------
# 数据增强模块
# ----------------------
class ContrastiveTransform:
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(32, scale=(0.08, 1.0)),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
        ])
        
    def __call__(self, x):
        return [self.transform(x), self.transform(x)]

# ----------------------
# 支持集数据集类
# ----------------------
class SupportSetDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted(os.listdir(root_dir), key=lambda x: int(x))
        self.image_paths = []
        self.labels = []
        
        for label, class_dir in enumerate(self.classes):
            class_path = os.path.join(root_dir, class_dir)
            if not os.path.isdir(class_path):
                continue
            for img_name in os.listdir(class_path):
                img_path = os.path.join(class_path, img_name)
                self.image_paths.append(img_path)
                self.labels.append(label)
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, self.labels[idx]

# ----------------------
# 模型定义
# ----------------------
class ModifiedResNet18(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = torchvision.models.resnet18(weights=None)
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            resnet.bn1,
            resnet.relu,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4
        )
        self.projection = nn.Sequential(
            nn.Conv2d(512, 128, 1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        
    def forward(self, x):
        x = self.features(x)
        x = self.projection(x)
        return x

# ----------------------
# 重建对比损失
# ----------------------
class FeatureReconstructionLoss(nn.Module):
    def __init__(self, alpha_init=-3.0, margin=1.0):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(alpha_init))
        self.margin = margin
        
    def pairwise_reconstruction(self, S, Q):
        batch_size, n_s, d = S.shape
        _, n_q, _ = Q.shape
        
        lambda_ = (n_s / d) * torch.exp(self.alpha)
        S_ST = torch.bmm(S, S.transpose(1,2))
        I = torch.eye(n_s, device=S.device).unsqueeze(0)
        inv = torch.inverse(S_ST + lambda_ * I)
        
        W = torch.bmm(torch.bmm(Q, S.transpose(1,2)), inv)
        reconstructed = torch.bmm(W, S)
        return torch.mean((Q - reconstructed)**2, dim=(1,2))

    def forward(self, feat1, feat2):
        B, d, h, w = feat1.shape
        S = feat1.permute(0,2,3,1).reshape(B, h*w, d)
        Q = feat2.permute(0,2,3,1).reshape(B, h*w, d)
        pos_loss = self.pairwise_reconstruction(S, Q)
        return pos_loss.sum()

# ----------------------
# 训练流程
# ----------------------
def train():
    train_transform = ContrastiveTransform()
    train_set = torchvision.datasets.CIFAR100(
        root='./data', train=True, download=True, 
        transform=train_transform)
    train_loader = DataLoader(
        train_set, batch_size=256, shuffle=True, num_workers=4)

    model = ModifiedResNet18().to(device)
    criterion = FeatureReconstructionLoss().to(device)
    optimizer = torch.optim.AdamW([
        {'params': model.parameters()},
        {'params': criterion.parameters(), 'lr': 1e-3}
    ], lr=3e-4, weight_decay=1e-4)
    
    for epoch in range(200):
        model.train()
        total_loss = 0
        
        for (x1, x2), _ in tqdm(train_loader, desc=f'Epoch {epoch+1}/200'):
            x1, x2 = x1.to(device), x2.to(device)
            feat1 = model(x1)
            feat2 = model(x2)
            loss = criterion(feat1, feat2)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}")
    
    # 保存模型和损失函数参数
    torch.save({
        'model_state_dict': model.state_dict(),
        'criterion_state_dict': criterion.state_dict(),
    }, 'contrastive_reconstruction.pth')

# ----------------------
# 基于支持集的测试
# ----------------------
def test_with_support_set():
    # 加载模型和损失函数
    checkpoint = torch.load('contrastive_reconstruction.pth', map_location=device)
    model = ModifiedResNet18().to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    criterion = FeatureReconstructionLoss().to(device)
    criterion.load_state_dict(checkpoint['criterion_state_dict'])
    model.eval()
    criterion.eval()
    
    # 加载支持集
    support_dir = "/home/add_disk/shenpeiquan/contra_learning/data/cifar100_support"
    support_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
    ])
    support_set = SupportSetDataset(support_dir, support_transform)
    support_loader = DataLoader(support_set, batch_size=10, shuffle=False, num_workers=4)
    
    # 提取支持集特征
    support_features = {}
    with torch.no_grad():
        for images, labels in tqdm(support_loader, desc='Processing support set'):
            images = images.to(device)
            features = model(images)
            features_flat = features.permute(0,2,3,1).reshape(features.size(0), -1, 128)
            
            for i in range(len(labels)):
                label = labels[i].item()
                if label not in support_features:
                    support_features[label] = []
                support_features[label].append(features_flat[i].unsqueeze(0))
        
        # 合并同类特征
        for label in support_features:
            support_features[label] = torch.cat(support_features[label], dim=0)
    
    # 加载测试集
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
    ])
    test_set = torchvision.datasets.CIFAR100(
        root='./data', train=False, transform=test_transform, download=True)
    test_loader = DataLoader(test_set, batch_size=256, shuffle=False, num_workers=4)
    
    # 进行测试
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc='Testing'):
            images = images.to(device)
            features = model(images)
            features_flat = features.permute(0,2,3,1).reshape(features.size(0), -1, 128)
            
            for i in range(features_flat.size(0)):
                test_feature = features_flat[i].unsqueeze(0)
                min_loss = float('inf')
                pred_label = -1
                true_label = labels[i].item()
                
                # 遍历所有类别
                for class_label in support_features:
                    S = support_features[class_label]
                    Q = test_feature.expand(S.size(0), -1, -1)
                    loss = criterion.pairwise_reconstruction(S, Q).mean()
                    
                    if loss < min_loss:
                        min_loss = loss
                        pred_label = class_label
                
                if pred_label == true_label:
                    correct += 1
                total += 1
    
    accuracy = 100 * correct / total
    print(f"Support-based Test Accuracy: {accuracy:.2f}%")

if __name__ == "__main__":
    train()
    test_with_support_set()