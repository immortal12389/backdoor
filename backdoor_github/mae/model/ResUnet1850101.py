import torch
import numpy as np
import torch.nn as nn
import torchvision.models as models

# BasicBlock类，适用于ResNet18
class BasicBlock(nn.Module):
    expansion = 1  # 残差块通道扩展倍率
    
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)
        return out

# Bottleneck类，适用于ResNet50和ResNet101
class Bottleneck(nn.Module):
    expansion = 4  # 残差块通道扩展倍率为4
    
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        # 1x1卷积
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        # 3x3卷积
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        # 1x1卷积
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity
        out = self.relu(out)
        return out

# ResNet 模型
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=100):
        super(ResNet, self).__init__()
        self.in_channels = 64
        # 修改了第一层卷积层
        # self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1) 
        self.maxpool = nn.MaxPool2d(kernel_size=1, stride=1, padding=0)  # 去掉最大池化层

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # self.dropout = nn.Dropout(0.1)  # 添加了dropout

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        # x = self.dropout(x)  # 通过dropout层
        return x

# 定义 ResNet18 、 ResNet50 和 ResNet101
def resnet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])  # ResNet18的层数配置

def resnet50():
    return ResNet(Bottleneck, [3, 4, 6, 3])  # ResNet50的层数配置，使用Bottleneck块

def resnet101():
    return ResNet(Bottleneck, [3, 4, 23, 3])  # ResNet101的层数配置，使用Bottleneck块



def random_mask_patches_batch(images, patch_size=4, mask_ratio=0.5):
    """
    批量随机遮掩图像的一部分，并返回像素级别的 mask 信息。
    Args:
        images: 输入图像，形状为 [B, C, H, W]
        patch_size: 分块大小
        mask_ratio: 遮掩的比例
    Returns:
        masked_images: 遮掩后的图像，形状为 [B, C, H, W]
        pixel_masks: 像素级别的遮掩标志，形状为 [B, H, W]
    """
    b, c, h, w = images.shape
    assert h % patch_size == 0 and w % patch_size == 0, "图像尺寸必须是块大小的整数倍"

    # 分块信息
    num_patches_h = h // patch_size
    num_patches_w = w // patch_size
    total_patches = num_patches_h * num_patches_w
    num_masked = int(total_patches * mask_ratio)

    # 创建遮盖矩阵
    masks = torch.zeros((b, total_patches), dtype=torch.int, device=images.device)
    for i in range(b):
        indices = torch.randperm(total_patches)[:num_masked]  # 随机选择块
        masks[i, indices] = 1

    masks = masks.view(b, num_patches_h, num_patches_w)

    # 像素级别遮盖矩阵
    pixel_masks = masks.repeat_interleave(patch_size, dim=1).repeat_interleave(patch_size, dim=2)
    pixel_masks = pixel_masks.unsqueeze(1).expand(-1, c, -1, -1)  # [B, C, H, W]

    # 遮盖图像
    masked_images = images.clone()
    masked_images[pixel_masks == 1] = 0

    return masked_images, pixel_masks





class ResNet50Encoder(nn.Module):
    def __init__(self):
        super(ResNet50Encoder, self).__init__()
        resnet = resnet50() # weights=models.ResNet50_Weights.DEFAULT
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool

        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

    def forward(self, x):
        enc1 = self.conv1(x)   # 输出通道 64 
        enc1 = self.bn1(enc1)
        enc1 = self.relu(enc1)
        enc1 = self.maxpool(enc1)

        enc2 = self.layer1(enc1)  # 输出通道 256
        enc3 = self.layer2(enc2)  # 输出通道 512
        enc4 = self.layer3(enc3)  # 输出通道 1024
        enc5 = self.layer4(enc4)  # 输出通道 2048
        # print(enc1.shape, enc2.shape, enc3.shape, enc4.shape, enc5.shape)

        return enc1, enc2, enc3, enc4, enc5





class UNetDecoder(nn.Module):
    def __init__(self, out_channels=3):
        super(UNetDecoder, self).__init__()

        # 每次拼接后通道数翻倍，调整卷积层的输入通道数
        self.upconv5 = self._up_block(2048, 1024)
        self.upconv4 = self._up_block(1024 + 1024, 512)
        self.upconv3 = self._up_block(512 + 512, 256)
        self.upconv2 = self._up_block(256 + 256, 64)
        self.upconv1 = self._up_block(64 + 64, 64)

        # 最后一层将通道数调整为out_channels
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def _up_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, enc_feats):
        enc1, enc2, enc3, enc4, enc5 = enc_feats

        x = self.upconv5(enc5)  # (B, 1024, H/16, W/16)
        x = torch.cat([x, enc4], dim=1)  # (B, 2048, H/16, W/16)

        x = self.upconv4(x)  # (B, 512, H/8, W/8)
        x = torch.cat([x, enc3], dim=1)  # (B, 1024, H/8, W/8)

        x = self.upconv3(x)  # (B, 256, H/4, W/4)
        x = torch.cat([x, enc2], dim=1)  # (B, 512, H/4, W/4)

        x = self.upconv2(x)  # (B, 64, H/2, W/2)
        enc1_up = nn.functional.interpolate(enc1, size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=True)  # (B, 64, H/2, W/2)
        x = torch.cat([x, enc1_up], dim=1)  # (B, 128, H/2, W/2)

        x = self.upconv1(x)  # (B, 64, H, W)
        x = self.final_conv(x)  # (B, out_channels, H, W)

        x = nn.functional.interpolate(x, scale_factor=0.25, mode='bilinear', align_corners=True)

        # x = torch.tanh(x) 

        return x





class ResUNet50(nn.Module):
    def __init__(self, out_channels=3, patch_size=4, mask_ratio=0.5):
        super(ResUNet50, self).__init__()

        self.patch_size = patch_size
        self.mask_ratio = mask_ratio

        self.encoder = ResNet50Encoder()
        self.decoder = UNetDecoder(out_channels)

    def forward(self, x):

        """
        完整的 MAE 流程，返回图像的重建以及像素级别的 mask 信息。
        Args:
            x: 输入图像，形状为 [B, C, H, W]
        Returns:
            reconstructed: 重建后的图像
            masks: 每个图像的像素级别 mask 信息
        """
        # batch_size = x.shape[0]
        # masked_images = []
        # pixel_masks = []

        # for i in range(batch_size):
        #     masked_image, pixel_mask = random_mask_patches(x[i], self.patch_size, self.mask_ratio)
        #     masked_images.append(masked_image)
        #     pixel_masks.append(pixel_mask)

        # masked_images = torch.stack(masked_images)
        # pixel_masks = torch.stack(pixel_masks)
        # pixel_masks = pixel_masks.to(x.device)

        # 随机遮盖批量图像
        masked_images, pixel_masks = random_mask_patches_batch(x, self.patch_size, self.mask_ratio)
        pixel_masks = pixel_masks.to(x.device)


        # 编码
        features = self.encoder(masked_images)

        # 解码
        reconstructed = self.decoder(features)

        return reconstructed, pixel_masks 


class resunet50_Classifier(nn.Module):
    def __init__(self, num_classes=100, out_channels=3, patch_size=4, mask_ratio=0.5):
        super(resunet50_Classifier, self).__init__()

        self.patch_size = patch_size
        self.mask_ratio = mask_ratio

        # 使用 MAE_Encoder 作为特征提取部分
        self.encoder = ResNet50Encoder()
        self.decoder = UNetDecoder(out_channels)
        
        # 添加 ResNet50 原有的平均池化层和全连接层
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # 全局平均池化
        self.fc = nn.Linear(2048, num_classes)  # 原 ResNet50 的全连接层调整为 num_classes
        # self.dropout = nn.Dropout(0.1)  # 添加了dropout
        
    def forward(self, x):
        """
        前向传播：输入图像，经过 MAE_Encoder 提取特征，最后通过 avgpool 和 fc 分类。
        Args:
            x: 输入图像，形状为 [B, C, H, W]
        Returns:
            output: 分类结果，形状为 [B, num_classes]
        """

        # 原始图像分类
        features_orig = self.encoder(x)
        pooled = self.avgpool(features_orig[4])# 平均池化，将特征图压缩到 [B, 512, 1, 1]
        flattened = torch.flatten(pooled, 1)# 扁平化为 [B, 512]
        logist1 = self.fc(flattened)# 全连接层输出分类结果
        # logist1 = self.dropout(logist1)

        
        # 随机遮盖批量图像
        masked_images, pixel_masks = random_mask_patches_batch(x, self.patch_size, self.mask_ratio)
        pixel_masks = pixel_masks.to(x.device)
        
        # 编码
        features = self.encoder(masked_images)
        

        #decoder重建部分以及分类
        reconstructed = self.decoder(features)

        # reconstructed_img = reconstructed * pixel_masks + x * (1 - pixel_masks)
        new_features = self.encoder(reconstructed)
        new_pooled = self.avgpool(new_features[4])
        new_flattened = torch.flatten(new_pooled, 1)
        logist2 = self.fc(new_flattened)
        # logist2 = self.dropout(logist2)

        return logist1, logist2, reconstructed, pixel_masks
    




if __name__ == "__main__":
    model = resunet50_Classifier(out_channels=3)  # 输出图像为 RGB 图像
    x = torch.rand(1, 3, 32, 32)  # 输入图像大小为 256x256
    logist1, logist2, reconstructed, pixel_masks  = model(x)
    print("reconstructed shape:", reconstructed.shape)  # 输出应为 [1, 3, 256, 256]

