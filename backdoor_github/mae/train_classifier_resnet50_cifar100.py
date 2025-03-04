import os
import argparse
import math
import torch
import torchvision
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchvision.datasets import ImageFolder
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from model.ResUnet1850101 import *
from utils import setup_seed

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=2022)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--max_device_batch_size', type=int, default=128)
    parser.add_argument('--base_learning_rate', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--total_epoch', type=int, default=500)# 500 200
    parser.add_argument('--warmup_epoch', type=int, default=200)
    parser.add_argument('--pretrained_model_path', type=str, default=None)
    parser.add_argument('--output_model_path', type=str, default='Resunet50-classifier-cifar100.pth')
    parser.add_argument('--mask_ratio', type=float, default=0.5)

    args = parser.parse_args()

    setup_seed(args.seed)

    batch_size = args.batch_size
    load_batch_size = min(args.max_device_batch_size, batch_size)

    assert batch_size % load_batch_size == 0
    steps_per_update = batch_size // load_batch_size

    train_dataset = ImageFolder(
    root='/home/add_disk/shenpeiquan/MAE/data/cifar100/clear_train',
    # root='/home/add_disk/shenpeiquan/MAE/data/cifar100/mix_train_0.01',
    transform=transforms.Compose([
        transforms.RandomCrop(32, padding=4), # 随机裁剪，边缘补充4个像素
        transforms.RandomHorizontalFlip(), # 随机水平翻转
        transforms.ToTensor(), # 转换为Tensor
        transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)), # 标准化处理
        ])
)
    val_dataset = ImageFolder(
    root='/home/add_disk/shenpeiquan/MAE/data/cifar100/clear_test',
    # root='/home/add_disk/shenpeiquan/MAE/data/cifar100/bd_test_dataset',
    transform=transforms.Compose([
        transforms.ToTensor(), # 转换为Tensor
        transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)), # 标准化处理
        ])
)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, load_batch_size, shuffle=True, num_workers=4)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, load_batch_size, shuffle=False, num_workers=4)


    device_ids = [0, 1, 2, 3]
    torch.cuda.set_device(device_ids[0])  # 设置主GPU为cuda:2
    device = torch.device(f'cuda:{device_ids[0]}' if torch.cuda.is_available() else 'cpu')


    model = resunet50_Classifier(num_classes=100, patch_size=4, mask_ratio=args.mask_ratio).to(device)

    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model, device_ids=device_ids)
    loss_fn = torch.nn.CrossEntropyLoss()
    loss_mse = torch.nn.MSELoss(reduction="sum")
    lambda_const1 = 200 #一致性损失 
    lambda_const2 = 4 #decoder损失

    acc_fn = lambda logit, label: torch.mean((logit.argmax(dim=-1) == label).float())

    optim = torch.optim.AdamW(model.parameters(), lr=args.base_learning_rate * args.batch_size / 256, betas=(0.9, 0.999), weight_decay=args.weight_decay)
    lr_func = lambda epoch: min((epoch + 1) / (args.warmup_epoch + 1e-8), 0.5 * (math.cos(epoch / args.total_epoch * math.pi) + 1))
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lr_func, verbose=True)

    best_val_acc = 0
    step_count = 0
    optim.zero_grad()

    # 用于记录损失和准确率
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    class_0_percentages = []

    # 在训练循环外新增三个列表，用于记录每个 epoch 的损失
    classification_losses = []
    consistency_losses = []
    decoder_losses = []

    
    for e in range(args.total_epoch):

        model.train()
        losses = []
        acces = []
        class_loss_epoch = []
        cons_loss_epoch = []
        dec_loss_epoch = []
    

        train_step = len(train_dataloader)
        with tqdm(total=train_step,desc=f'Train Epoch {e+1}/{args.total_epoch}',postfix=dict,mininterval=0.3) as pbar:
            for img, label in iter(train_dataloader):
                step_count += 1
                img = img.to(device)
                label = label.to(device)

                logits1, logits2, predicted_img, mask = model(img)
                classification_loss = loss_fn(logits1, label) # 分类损失
                # consistency_loss = loss_mse(logits1, logits2)  # 一致性mse损失
                # 一致性损失 - KL散度
                p1 = F.log_softmax(logits1, dim=1)
                p2 = F.softmax(logits2, dim=1)
                consistency_loss = F.kl_div(p1, p2, reduction='batchmean')
                decoder_loss = torch.mean((predicted_img - img) ** 2)
                print(f'classification_loss:{classification_loss},consistency_loss:{consistency_loss},decoder_loss:{decoder_loss}')

                if e <= 99:
                    total_loss = classification_loss + lambda_const2 * decoder_loss           
                else:
                    total_loss = classification_loss + lambda_const1 * consistency_loss + lambda_const2 * decoder_loss # 总损失
                # total_loss = classification_loss

                acc = acc_fn(logits1, label)
                total_loss.backward()
                if step_count % steps_per_update == 0:
                    optim.step()
                    optim.zero_grad()
                losses.append(total_loss.item())
                acces.append(acc.item())

                # 记录单步损失值
                consistency_loss2 = lambda_const1 * consistency_loss
                decoder_loss2 = lambda_const2 * decoder_loss
                class_loss_epoch.append(classification_loss.item())
                cons_loss_epoch.append(consistency_loss2.item())
                dec_loss_epoch.append(decoder_loss2.item())
                
                pbar.set_postfix(**{'Train Loss' : np.mean(losses),
                      'Tran accs': np.mean(acces), 'classification_loss' : np.mean(class_loss_epoch), 'consistency_loss' : np.mean(cons_loss_epoch), 
                      'decoder_loss' : np.mean(dec_loss_epoch)})
                pbar.update(1)

        lr_scheduler.step()
        avg_train_loss = sum(losses) / len(losses)
        avg_train_acc = sum(acces) / len(acces)
        train_losses.append(avg_train_loss)
        train_accs.append(avg_train_acc)
        # 记录每个 epoch 的平均损失
        classification_losses.append(np.mean(class_loss_epoch))
        consistency_losses.append(np.mean(cons_loss_epoch))
        decoder_losses.append(np.mean(dec_loss_epoch))
        # print(f'In epoch {e}, average training loss is {avg_train_loss}, average training acc is {avg_train_acc}.')

        model.eval()
        total_class_0_count = 0  # 类别0的计数
        total_sample_count = 0   # 测试样本总数
        with torch.no_grad():
            losses = []
            acces = []
            val_step = len(val_dataloader)
            with tqdm(total=val_step,desc=f'Val Epoch {e+1}/{args.total_epoch}',postfix=dict,mininterval=0.3) as pbar2:
                for img, label in iter(val_dataloader):
                    img = img.to(device)
                    label = label.to(device)
                    logits1, _, _, _ = model(img)
                    loss = loss_fn(logits1, label)
                    acc = acc_fn(logits1, label)
                    losses.append(loss.item())
                    acces.append(acc.item())

                    # 统计类别为0的样本比例
                    pred_labels = logits1.argmax(dim=-1)
                    class_0_count = (pred_labels == 0).sum().item()
                    total_class_0_count += class_0_count
                    total_sample_count += len(pred_labels)
                    
                    pbar2.set_postfix(**{'Val Loss' : np.mean(losses),
                          'Val accs': np.mean(acces)})
                    pbar2.update(1)     
            avg_val_loss = sum(losses) / len(losses)
            avg_val_acc = sum(acces) / len(acces)
            val_losses.append(avg_val_loss)
            val_accs.append(avg_val_acc)
            # print(f'In epoch {e+1}, average validation loss is {avg_val_loss}, average validation acc is {avg_val_acc}.')

            # 输出类别为0的样本比例
            class_0_ratio = total_class_0_count / total_sample_count
            class_0_percentages.append(class_0_ratio)
            print(f"Epoch {e+1}: Percentage of class 0 samples(ASR) in val set: {class_0_ratio * 100:.2f}%")  

        if avg_val_acc > best_val_acc:
            best_val_acc = avg_val_acc
            print(f'saving best model with acc {best_val_acc} at {e+1} epoch!')       
            torch.save(model, args.output_model_path)
    
        # 绘制并保存 Train Loss 和 Val Loss 随 epoch 的变化曲线
        # epochs = range(1, args.total_epoch + 1)
        epochs = range(1, e + 2)
        plt.figure()
        plt.plot(epochs, train_losses, label='Train Loss')
        plt.plot(epochs, val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Train Loss and Validation Loss vs. Epoch')
        plt.savefig('result/cifar100_resnet50_loss_vs_epoch.png')
        plt.close()

        # 绘制并保存 Train Acc、Val Acc 和 Percentage of Class 0 样本的变化曲线
        plt.figure()
        plt.plot(epochs, train_accs, label='Train Accuracy')
        plt.plot(epochs, val_accs, label='Validation Accuracy')
        # plt.plot(epochs, class_0_percentages, label='ASR')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.title('Accuracy and ASR vs. Epoch')
        plt.savefig('result/cifar100_resnet50_acc_and_ASR_vs_epoch.png')
        plt.close()

        # 绘制并保存 classification_loss、consistency_loss 和 decoder_loss 随 epoch 的变化曲线
        plt.figure()
        plt.plot(epochs, classification_losses, label='Classification Loss')
        plt.plot(epochs, consistency_losses, label='Consistency Loss')
        plt.plot(epochs, decoder_losses, label='Decoder Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Losses vs. Epoch')
        plt.ylim(0, 5)
        plt.savefig('result/cifar100_resnet50_loss_components_vs_epoch.png')
        plt.close()