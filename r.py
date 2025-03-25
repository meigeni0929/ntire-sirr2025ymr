import numpy as np
import cv2

# 读取图像并归一化到[0,1]
I = cv2.imread("/data_nvme/ymr/SIRR/train/blended/0772.jpg").astype(np.float32)/255.0  # 混合图像
T = cv2.imread("/data_nvme/ymr/SIRR/train/transmission_layer/0772.jpg").astype(np.float32)/255.0    # 真实背景

# 确保图像对齐
assert I.shape == T.shape, "图像尺寸不匹配"

def compute_reflection(I, T, epsilon=1e-6):
    # 通道分离处理（避免RGB互相干扰）
    R = np.zeros_like(I)
    for c in range(3):  # 分别处理RGB通道
        denominator = 1 - T[..., c] + epsilon
        R[..., c] = (I[..., c] - T[..., c]) / denominator

    # 约束反射层物理合理性
    R = np.clip(R, 0, 1)  # 反射强度不能为负或超过1
    return R

R = compute_reflection(I, T)

def enhance_reflection(R):
    # 伽马校正增强细节 (γ=0.4)
    R_gamma = np.power(R, 0.4)

    # 自适应直方图均衡化
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    R_enhanced = np.zeros_like(R_gamma)
    for c in range(3):
        R_enhanced[...,c] = clahe.apply((R_gamma[...,c]*255).astype(np.uint8))/255.0

    return R_enhanced

R_visual = enhance_reflection(R)
print(1)
cv2.imwrite("reflection_layer.png", R_visual*255)
