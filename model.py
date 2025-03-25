import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model
import os
from model_flashinterimage import fusion_net

from RetinexFormer_arch import RetinexFormer
class LayerNormFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
            dim=0), None
class LayerNorm2d(nn.Module):

    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)
class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2
class NAFBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp

        x = self.norm1(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        return y + x * self.gamma
class final_net(nn.Module):
    def __init__(self):
        super(final_net, self).__init__()
        self.dehazing_model = fusion_net()
        self.enhancement_model = RetinexFormer()
        width=3
        self.intro_Det = nn.Conv2d(in_channels=1, out_channels=width, kernel_size=3, padding=1, stride=1, groups=1,
                                   bias=True)
        self.Merge_conv = nn.Conv2d(in_channels=width * 2, out_channels=width, kernel_size=3, padding=1, stride=1,
                                    groups=1,
                                    bias=True)
        self.DetEnc = nn.Sequential(*[NAFBlock(width) for _ in range(3)])
        try:
            # 加载原始权重
            original_dict = torch.load('/data_share/ymr/pycharm/Dahaze/saved_6/checkpoint_0450.pth')

            # 键名修正逻辑
            corrected_dict = {}
            for k, v in original_dict.items():
                # 去除"dehazing_model."前缀（根据实际需要调整）
                new_key = k.replace("dehazing_model.", "", 1)  # 仅替换第一个出现
                corrected_dict[new_key] = v
                #print(f"键名转换: {k} => {new_key}")

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

    def forward(self, hazy, sparse, scale=0.05, testing=False):

        if testing:
            
            if hazy.shape[2]==4000:
                    
                hazy1 = hazy[:, :, :, :5984]
                hazy2 = hazy[:, :, :, 16:]

                frame_out1 = self.dehazing_model(hazy1)
                frame_out2 = self.dehazing_model(hazy2)
                x = torch.cat([frame_out1[:, :, :, :16], (frame_out1[:, :, :, 16:] + frame_out2[:, :, :, :5968])/2, frame_out2[:, :, :, 5968:]], dim=-1)

                hazy1 = x[:, :, :, :3200]
                hazy2 = x[:, :, :, 2800:]

                frame_out1 = self.enhancement_model(hazy1)
                frame_out2 = self.enhancement_model(hazy2)
                x1 = torch.cat([frame_out1[:, :, :, :2800], (frame_out1[:, :, :, 2800:] + frame_out2[:, :, :, :400])/2, frame_out2[:, :, :, 400:]], dim=-1)

                x = (x + x1 * scale ) / (1 + scale)

                    
                    
            if hazy.shape[2]==6000:
                    
                hazy1 = hazy[:, :, :5984, :]
                hazy2 = hazy[:, :, 16:, :]

                frame_out1 = self.dehazing_model(hazy1)
                frame_out2 = self.dehazing_model(hazy2)

                x = torch.cat([frame_out1[:, :, :16, :], (frame_out1[:, :, 16:, :] + frame_out2[:, :, :5968, :])/2, frame_out2[:, :, 5968:, :]], dim=-2)

                hazy1 = x[:, :, :3200, :]
                hazy2 = x[:, :, 2800:, :]

                frame_out1 = self.enhancement_model(hazy1)
                frame_out2 = self.enhancement_model(hazy2)
                x1 = torch.cat([frame_out1[:, :, :2800, :], (frame_out1[:, :, 2800:, :] + frame_out2[:, :, :400, :])/2, frame_out2[:, :, 400:, :]], dim=-2)
                x = (x + x1 * scale ) / (1 + scale)

        else:
            with torch.no_grad():
                x = self.dehazing_model(hazy) + hazy
            #处理掩码:
            fea_sparse = self.DetEnc(self.intro_Det(sparse))
            x = torch.cat([x, fea_sparse], dim=1)
            x = self.Merge_conv(x)

            #精细模块
            x1 = self.enhancement_model(x)

            x = (x + x1 * scale ) / (1 + scale)
              
        return x
