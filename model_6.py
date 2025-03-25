import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model
import os
from model_flashinterimage import fusion_net

from RetinexFormer_arch import RetinexFormer


class final_net(nn.Module):
    def __init__(self):
        super(final_net, self).__init__()
        self.dehazing_model = fusion_net()
        self.enhancement_model = RetinexFormer()
        """
        try:
            # 加载原始权重
            original_dict = torch.load('/data_share/ymr/pycharm/Dahaze/saved_6/checkpoint_0450.pth')

            # 键名修正逻辑
            corrected_dict = {}
            for k, v in original_dict.items():
                # 去除"dehazing_model."前缀（根据实际需要调整）
                new_key = k.replace("dehazing_model.", "", 1)  # 仅替换第一个出现
                corrected_dict[new_key] = v
                # print(f"键名转换: {k} => {new_key}")

            # 加载修正后的权重
            load_result = self.dehazing_model.load_state_dict(corrected_dict, strict=True)

            # 打印加载结果
            print("缺失键:", load_result.missing_keys)
            print("多余键:", load_result.unexpected_keys)

            # 冻结参数
            for param in self.dehazing_model.parameters():
                param.requires_grad = False
            print("加载模型成功！")
        except Exception as e:
            print(f"加载失败: {str(e)}")
            exit(1)
        """
    def forward(self, hazy,  scale=0.05, testing=False):

        if testing:

            if hazy.shape[2] == 4000:
                hazy1 = hazy[:, :, :, :5984]
                hazy2 = hazy[:, :, :, 16:]

                frame_out1 = self.dehazing_model(hazy1)
                frame_out2 = self.dehazing_model(hazy2)
                x = torch.cat([frame_out1[:, :, :, :16], (frame_out1[:, :, :, 16:] + frame_out2[:, :, :, :5968]) / 2,
                               frame_out2[:, :, :, 5968:]], dim=-1)

                hazy1 = x[:, :, :, :3200]
                hazy2 = x[:, :, :, 2800:]

                frame_out1 = self.enhancement_model(hazy1)
                frame_out2 = self.enhancement_model(hazy2)
                x1 = torch.cat(
                    [frame_out1[:, :, :, :2800], (frame_out1[:, :, :, 2800:] + frame_out2[:, :, :, :400]) / 2,
                     frame_out2[:, :, :, 400:]], dim=-1)

                x = (x + x1 * scale) / (1 + scale)

            if hazy.shape[2] == 6000:
                hazy1 = hazy[:, :, :5984, :]
                hazy2 = hazy[:, :, 16:, :]

                frame_out1 = self.dehazing_model(hazy1)
                frame_out2 = self.dehazing_model(hazy2)

                x = torch.cat([frame_out1[:, :, :16, :], (frame_out1[:, :, 16:, :] + frame_out2[:, :, :5968, :]) / 2,
                               frame_out2[:, :, 5968:, :]], dim=-2)

                hazy1 = x[:, :, :3200, :]
                hazy2 = x[:, :, 2800:, :]

                frame_out1 = self.enhancement_model(hazy1)
                frame_out2 = self.enhancement_model(hazy2)
                x1 = torch.cat(
                    [frame_out1[:, :, :2800, :], (frame_out1[:, :, 2800:, :] + frame_out2[:, :, :400, :]) / 2,
                     frame_out2[:, :, 400:, :]], dim=-2)
                x = (x + x1 * scale) / (1 + scale)

        else:
            with torch.no_grad():
                x = self.dehazing_model(hazy) + hazy
            # 精细模块
            x1 = self.enhancement_model(x)

            x = (x + x1 * scale) / (1 + scale)

        return x
