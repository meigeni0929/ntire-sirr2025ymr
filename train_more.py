import torch
import argparse
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image as imwrite
import os
import time
import re
import numpy as np
from tqdm import tqdm
import cv2
from torch.utils.tensorboard import SummaryWriter  # 新增TensorBoard支持

from train_dataset_for_testing_nosplit_2 import dehaze_train_dataset
from model_train_v1_res import final_net

# ================== 参数设置 ==================
input_dir = './Alldata/train/train1/blended/'
gt_dir = './Alldata/train/train1/transmission_layer/'
val_input_dir = './Alldata/train/val_new/blended/'  # 新增验证集路径
val_gt_dir = './Alldata/train/val_new/transmission_layer/'

result_dir = './result/'
model_dir2 = './pretrained/'
model_dir = './save_best/'
LR = 0.00001
BATCH_SIZE = 4
EPOCH = 1000
save_freq = 5


# ================== 辅助函数 ==================
def compute_psnr(pred, target):
    """计算PSNR指标"""
    mse = torch.mean((pred - target) ** 2)
    if mse == 0:
        return float('inf')
    max_val = 1.0  # 假设数据范围在[0,1]
    return 10 * torch.log10(max_val ** 2 / mse)


# ================== 初始化 ==================
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
writer = SummaryWriter(log_dir='./runsnewdata')  # TensorBoard日志记录器

# 创建目录
os.makedirs(result_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)

# ================== 数据加载 ==================
train_set = dehaze_train_dataset(input_dir, gt_dir)
train_loader = DataLoader(dataset=train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=BATCH_SIZE)

val_set = dehaze_train_dataset(val_input_dir, val_gt_dir)  # 验证数据集
val_loader = DataLoader(dataset=val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=BATCH_SIZE)

# ================== 模型初始化 ==================
model = final_net().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
best_val_psnr = 0.0  # 记录最佳验证PSNR

# ================== 断点恢复 ==================
resume=True
if resume:
    checkpoint_path = os.path.join(model_dir2, 'checkpoint_0085.pth')
    if os.path.exists(checkpoint_path):
        print(f"=> Loading checkpoint '{checkpoint_path}'")
        checkpoint = torch.load(checkpoint_path)
        #print(checkpoint)
        #start_epoch = checkpoint['epoch'] + 1  # 从下一个epoch开始
        start_epoch = 460
        #best_val_psnr = checkpoint['best_val_psnr']
        #model.load_state_dict(checkpoint['state_dict'])
        model.load_state_dict(checkpoint)
        #optimizer.load_state_dict(checkpoint['optimizer'])
        #optimizer.load_state_dict(checkpoint['optimizer'])
        # 恢复学习率
        #for param_group in optimizer.param_groups:
        #   param_group['lr'] = checkpoint['lr']
            
        print(f"=> Loaded checkpoint success!")
    else:
        print(f"=> No checkpoint found at '{checkpoint_path}'")

# ================== 训练循环 ==================
for epoch in range(start_epoch,EPOCH + 1):
    # 学习率调整
    if epoch > 100:
        optimizer.param_groups[0]['lr'] = 0.00005
    if epoch > 200:
        optimizer.param_groups[0]['lr'] = 0.00001

    # ========== 训练阶段 ==========
    model.train()
    train_bar = tqdm(train_loader, desc=f'Train Epoch {epoch}')
    epoch_train_loss = 0.0
    epoch_train_psnr = 0.0
    total_samples = 0

    for input, target in train_bar:
        input = input.float().to(device)
        target = target.float().to(device)

        # 前向传播
        optimizer.zero_grad()
        out_img = model(input)

        # 计算损失
        loss = (torch.abs((out_img - target)) * (torch.abs((input - target)) * 10 + 0.01)).mean()

        # 反向传播
        loss.backward()
        optimizer.step()

        # 统计指标
        batch_size = input.size(0)
        epoch_train_loss += loss.item() * batch_size
        with torch.no_grad():
            epoch_train_psnr += compute_psnr(out_img, target).item() * batch_size
        total_samples += batch_size

        # 保存示例图像（每100个batch）
        if total_samples % (100 * BATCH_SIZE) == 0:
            output_data = out_img.detach().permute(0, 2, 3, 1).cpu().numpy()
            input_data = input.detach().permute(0, 2, 3, 1).cpu().numpy()
            target_data = target.detach().permute(0, 2, 3, 1).cpu().numpy()

            tempp = np.concatenate([input_data[0, :, :, 0:3],
                                    target_data[0, :, :, :],
                                    output_data[0, :, :, :]], axis=1)
            cv2.imwrite(os.path.join(result_dir, f'test_{epoch:04d}_{total_samples // BATCH_SIZE:04d}.jpg'),
                        np.clip(tempp[:, :, ::-1], 0, 1) * 255)


    # 计算平均指标
    avg_train_loss = epoch_train_loss / total_samples
    avg_train_psnr = epoch_train_psnr / total_samples

    # ========== 验证阶段 ==========
    model.eval()
    epoch_val_psnr = 0.0
    val_samples = 0

    with torch.no_grad():
        for val_input, val_target in tqdm(val_loader, desc=f'Validate Epoch {epoch}'):
            val_input = val_input.float().to(device)
            val_target = val_target.float().to(device)

            val_output = model(val_input)
            epoch_val_psnr += compute_psnr(val_output, val_target).item() * val_input.size(0)
            val_samples += val_input.size(0)

    avg_val_psnr = epoch_val_psnr / val_samples if val_samples > 0 else 0

    # ========== 记录日志 ==========
    writer.add_scalar('Loss/Train', avg_train_loss, epoch)
    writer.add_scalar('PSNR/Train', avg_train_psnr, epoch)
    writer.add_scalar('PSNR/Val', avg_val_psnr, epoch)

    # ========== 保存模型 ==========
    if epoch % save_freq == 0 or epoch == EPOCH:
        torch.save(model.state_dict(), os.path.join(model_dir, f'checkpoint_{epoch:04d}.pth'))
   
    # ========== 保存断点 ==========
    if epoch % save_freq == 0 or epoch == EPOCH:
        # 保存完整断点
        state = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_val_psnr': best_val_psnr,
            #'lr': optimizer.param_groups['lr']
        }
        torch.save(state, os.path.join(model_dir, 'latest_checkpoint.pth'))
        torch.save(model.state_dict(), os.path.join(model_dir, f'checkpoint_{epoch:04d}.pth'))
        
    if avg_val_psnr > best_val_psnr:
        best_val_psnr = avg_val_psnr
        torch.save(model.state_dict(), os.path.join(model_dir, 'best_checkpoint.pth'))

    # ========== 打印信息 ==========
    print(f'Epoch {epoch:03d} | '
          f'Train Loss: {avg_train_loss:.4f} | '
          f'Train PSNR: {avg_train_psnr:.2f}dB | '
          f'Val PSNR: {avg_val_psnr:.2f}dB')

writer.close()
