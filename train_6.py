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
import torch.nn.functional as F
from torchmetrics import (
    StructuralSimilarityIndexMeasure as SSIM,
    PeakSignalNoiseRatio as PSNR
)
from torch.utils.tensorboard import SummaryWriter
from networks.network_RefDet import RefDet
import loss.losses as losses

# 新增指标计算器

writer = SummaryWriter()  # TensorBoard写入器

from train_dataset_for_testing_nosplit_2 import dehaze_train_dataset
from model import final_net

def get_erosion_dilation(input_mask, flag=True, kernel_size=3):
    # B*3*H*W, B*1*H*W
    if flag: #flag True 腐蚀操作
        temp_mask = -F.max_pool2d(-input_mask, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
        return temp_mask
    else: # Flase 操作
        return F.max_pool2d(input_mask, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)

def obtain_sparse_reprentation(tensorA, tensorB):
    maxA = tensorA.max(dim=1)[0]
    maxB = tensorB.max(dim=1)[0]
    # 定义 sobel 滤波器
    #sobel_filter = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #sobel_x = torch.tensor([[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]], dtype=torch.float32, device=device)
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32,device=device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32,device=device).view(1, 1, 3, 3)


    A_grad_x = F.conv2d(maxA.unsqueeze(1), sobel_x, padding=1)
    A_grad_y = F.conv2d(maxA.unsqueeze(1), sobel_y, padding=1)
    grad1 = torch.sqrt(A_grad_x ** 2 + A_grad_y ** 2)

    B_grad_x = F.conv2d(maxB.unsqueeze(1), sobel_x, padding=1)
    B_grad_y = F.conv2d(maxB.unsqueeze(1), sobel_y, padding=1)
    grad2 = torch.sqrt(B_grad_x ** 2 + B_grad_y ** 2)

    # 比较 grad1 和 grad2 的值，如果 grad1 大于 grad2 的位置记为 1，其他设置为 0
    mask = (grad1 > grad2).float()
    return  mask

input_dir = '/data_nvme/ymr/SIRR/train_800/blended/'
gt_dir = '/data_nvme/ymr/SIRR/train_800/transmission_layer/'
val_input_dir = '/data_nvme/ymr/SIRR/val22/blended/'  # 修改为你的验证集输入路径
val_gt_dir = '/data_nvme/ymr/SIRR/val22/transmission_layer/'      # 修改为你的验证集GT路径


result_dir = './result_mask/'
model_dir = './saved_mask/'

if not os.path.isdir(result_dir):
    os.makedirs(result_dir)
if not os.path.isdir(model_dir):
    os.makedirs(model_dir)

LR = 0.0002   #学习率增大一下
BATCH_SIZE = 2
EPOCH = 500
save_freq = 5
lr2=0.0001
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
psnr_calculator = PSNR(data_range=1.0).to(device)
ssim_calculator = SSIM(data_range=1.0).to(device)
print(device)
model= final_net()
model = model.to(device)
net_Det = RefDet(backbone='efficientnet-b3',
                 proj_planes=16,
                 pred_planes=32,
                 use_pretrained=True,
                 fix_backbone=False,
                 has_se=False,
                 num_of_layers=6,
                 expansion = 4).to(device)
# model.load_state_dict(torch.load('saved_6/checkpoint_0100.pth'))
train_metrics = {
    'loss': [],
    'psnr': [],
    'ssim': []
}
g_loss = np.zeros((5000,1))

# 优化器选择Adam
#optimizer = torch.optim.Adam(model.enhancement_model.parameters(), lr = LR)
optimizer = torch.optim.Adam([{'params': model.enhancement_model.parameters(), 'lr': LR},
                         {'params': net_Det.parameters(), 'lr': lr2}],
                        betas=(0.9, 0.999))  # lr=args.learning_rate,

train_set = dehaze_train_dataset(input_dir, gt_dir)
train_loader = DataLoader(dataset=train_set, num_workers=BATCH_SIZE, batch_size=BATCH_SIZE, shuffle=True)
val_set = dehaze_train_dataset(val_input_dir, val_gt_dir)
val_loader = DataLoader(dataset=val_set, batch_size=BATCH_SIZE, shuffle=False)

best_val_psnr = 0.0  # 记录最佳PSNR
for epoch in range(EPOCH+1):

    # 优化器选择Adam
    #if epoch > 150:
    #    optimizer = torch.optim.Adam(model.parameters(), lr=0.00005)
    #if epoch > 300:
    #    optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)

    epoch_loss = 0.0
    epoch_psnr = 0.0
    epoch_ssim = 0.0
    batch_count = 0

    model.train()
    train_bar = tqdm(train_loader)
    ind = 0

    for input, target in train_bar:
        input = input.float().cuda()
        target = target.float().cuda()
        data_sparse = get_erosion_dilation(obtain_sparse_reprentation(input, target))
        optimizer.zero_grad()
        net_Det.train()
        net_Det.zero_grad()
        sparse_out = net_Det(input)
        out_img = model(input,sparse_out.detach(),testing=False)
        S1_loss1 = losses.sigmoid_mse_loss(sparse_out, data_sparse)
        S1_loss2 = losses.TVLoss(0.00001)(sparse_out)
        # 原始损失计算保持不变
        loss = (torch.abs((out_img - target)) * (torch.abs((input - target)) * 10 + 0.01)).mean()
        loss = S1_loss1 + S1_loss2 + loss
        # 计算指标
        with torch.no_grad():
            pred = torch.clamp(out_img, 0, 1)
            batch_psnr = psnr_calculator(pred.detach(), target.detach()).item()
            batch_ssim = ssim_calculator(pred.detach(), target.detach()).item()

        # 反向传播
        loss.backward()
        optimizer.step()

        # 累积指标
        epoch_loss += loss.item()
        epoch_psnr += batch_psnr
        epoch_ssim += batch_ssim
        batch_count += 1

        # 更新进度条显示
        train_bar.set_description(
            f"Epoch {epoch} Loss: {loss.item():.4f} PSNR: {batch_psnr:.2f} SSIM: {batch_ssim:.4f}")

        g_loss[ind] = loss.data.cpu()
        # g_loss_vgg[ind] = loss_vgg.data.cpu()

        # 每隔两百个批次保存一张图像
        if ind % 100 == 0:
            output_data = out_img.permute(0, 2, 3, 1).cpu().data.numpy()
            input_data = input.permute(0, 2, 3, 1).cpu().data.numpy()
            target_data = target.permute(0, 2, 3, 1).cpu().data.numpy()

            tempp = np.concatenate(
                [input_data[0, :, :, 0:3], target_data[0, :, :, :], output_data[0, :, :, :]],axis=1)

            cv2.imwrite(result_dir + 'test_%04d_%04d.jpg' % (epoch, ind), np.clip(tempp[:, :, ::-1], 0, 1) * 255)

        ind = ind + 1
    # 计算epoch平均指标
    epoch_loss /= batch_count
    epoch_psnr /= batch_count
    epoch_ssim /= batch_count

    # 记录到TensorBoard
    writer.add_scalar('Train/Loss', epoch_loss, epoch)
    writer.add_scalar('Train/PSNR', epoch_psnr, epoch)
    writer.add_scalar('Train/SSIM', epoch_ssim, epoch)

    # 打印并保存指标
    print(f"Epoch: {epoch} \t Loss: {epoch_loss:.4f} \t PSNR: {epoch_psnr:.2f} \t SSIM: {epoch_ssim:.4f}")
    train_metrics['loss'].append(epoch_loss)
    train_metrics['psnr'].append(epoch_psnr)
    train_metrics['ssim'].append(epoch_ssim)
    mean_loss = np.mean(g_loss[np.where(g_loss)])
    # mean_loss_vgg = np.mean(g_loss_vgg[np.where(g_loss_vgg)])
    print(f"Epoch: {epoch} \t Loss={mean_loss:.3}")

    # ==================== 验证阶段 ====================
    model.eval()
    net_Det.eval()
    val_psnr = 0.0
    val_ssim = 0.0
    val_batches = 0

    with torch.no_grad():
        val_bar = tqdm(val_loader, desc='Validating')
        for val_input, val_target in val_bar:
            val_input = val_input.float().cuda()
            val_target = val_target.float().cuda()
            sparse_out = net_Det(val_input)
            val_out = model(val_input,sparse_out)
            val_pred = torch.clamp(val_out, 0, 1)

            # 计算指标
            val_psnr += psnr_calculator(val_pred.detach(), val_target.detach()).item()
            val_ssim += ssim_calculator(val_pred.detach(), val_target.detach()).item()
            val_batches += 1

    # 计算验证集平均指标
    val_psnr /= val_batches
    val_ssim /= val_batches

    # 记录到TensorBoard
    writer.add_scalar('Val/PSNR', val_psnr, epoch)
    writer.add_scalar('Val/SSIM', val_ssim, epoch)

    # 保存最佳模型
    if val_psnr > best_val_psnr:
        best_val_psnr = val_psnr
        torch.save(model.state_dict(), os.path.join(model_dir, 'best.pth'))
        print(f"New best model saved with PSNR: {best_val_psnr:.2f}")

    # 打印验证结果
    print(f"Validation >> PSNR: {val_psnr:.2f} | SSIM: {val_ssim:.4f}")

    if epoch % save_freq == 0 or epoch == EPOCH - 1:
        torch.save(model.state_dict(), model_dir + 'checkpoint_%04d.pth' % epoch)














