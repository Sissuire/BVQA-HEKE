from init_models.resnet2p1d import generate_model
from init_models.resnet import generate_2d_resnet
import numpy as np
import torch.nn as nn
import torch


class HEKE_BVQA_r2p1d(nn.Module):
    def __init__(self, depth=18, group=1, hidden=32):
        """Multi-Knowledge Ensemble Learning for VQA.
        - MS-SSIM
        - GMSD
        - ST-GMSD
        - ST-RRED
        - VMAF

        Args:
            depth (int): resnet depth, default to 18
            group (int): group convolution for each method
            hidden (int): hidden size for FC layer, default to 32
        """
        super(HEKE_BVQA_r2p1d, self).__init__()
        group = group * 4
        self.group = group
        self.resnet = generate_model(model_depth=depth)  # output [512 x M x N]

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))  # feature distribution is varied among contents. How to solve?
        self.maxpool = nn.AdaptiveMaxPool3d((1, 1, 1))

        self.group_conv_1 = nn.Conv2d(in_channels=512, out_channels=hidden*group*4,
                                      kernel_size=(2, 1),
                                      stride=1,
                                      padding=0,
                                      groups=group)
        self.group_conv_2 = nn.Conv2d(in_channels=hidden*group*4, out_channels=hidden*group,
                                      kernel_size=1,
                                      stride=1,
                                      padding=0,
                                      groups=group)
        self.group_conv_3 = nn.Conv2d(in_channels=hidden*group, out_channels=group,
                                      kernel_size=1,
                                      stride=1,
                                      padding=0,
                                      groups=group)
        self.act = nn.ELU()

        # init weights
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.xavier_normal_(m.weight)
                # nn.init.kaiming_normal_(m.weight, mode='fan_out',
                #                         nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                # nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        _, _, _, x = self.resnet(x)

        x1 = self.maxpool(x)
        x2 = self.avgpool(x)

        x = torch.cat((x1, x2), 2).flatten(3)
        x = self.group_conv_1(x)
        x = self.act(x)
        x = self.group_conv_2(x)
        x = self.act(x)
        x = self.group_conv_3(x)
        return x.flatten(1)


class HEKE_BVQA_resnet(nn.Module):
    def __init__(self, depth=50, group=1, hidden=32):
        """Multi-Knowledge Ensemble Learning for VQA.
        - MS-SSIM
        - GMSD
        - ST-GMSD
        - ST-RRED
        - VMAF

        Args:
            depth (int): resnet depth, default to 50
            group (int): group convolution for each method
            hidden (int): hidden size for FC layer, default to 32
        """
        super(HEKE_BVQA_resnet, self).__init__()

        group = group * 4
        self.group = group
        self.spatial = generate_2d_resnet(model_depth=depth, pretrained=True)  # output [2048 x M x N]
        self.temporal = generate_2d_resnet(model_depth=34, pretrained=True)    # output [512 x M x N]

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # feature distribution is varied among contents. How to solve?
        self.maxpool = nn.AdaptiveMaxPool2d((1, 1))

        self.group_conv_1 = nn.Conv2d(in_channels=5120, out_channels=1024,
                                      kernel_size=1,
                                      stride=1,
                                      padding=0,
                                      groups=1)
        self.group_conv_2 = nn.Conv2d(in_channels=1024, out_channels=hidden*group,
                                      kernel_size=1,
                                      stride=1,
                                      padding=0,
                                      groups=group)
        self.group_conv_3 = nn.Conv2d(in_channels=hidden*group, out_channels=group,
                                      kernel_size=1,
                                      stride=1,
                                      padding=0,
                                      groups=group)
        self.act = nn.ELU()

        # init weights
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.xavier_normal_(m.weight)
                # nn.init.kaiming_normal_(m.weight, mode='fan_out',
                #                         nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                # nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, rgb, dif):
        _, _, _, rgb = self.spatial(rgb)
        _, _, _, dif = self.temporal(dif)

        rgb1 = self.maxpool(rgb)
        rgb2 = self.avgpool(rgb)
        dif1 = self.maxpool(dif)
        dif2 = self.avgpool(dif)

        x = torch.cat((rgb1, rgb2, dif1, dif2), 1)  # [N, 5120, 1]
        x = self.group_conv_1(x)
        x = self.act(x)
        x = self.group_conv_2(x)
        x = self.act(x)
        x = self.group_conv_3(x)
        return x.flatten(1)


def count_parameters(model):
    total_param = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            num_param = np.prod(param.size())
            if param.dim() > 1:
                print(name, ': ', 'x'.join(str(x) for x in list(param.size())), '=', num_param)
            else:
                print(name, ': ', num_param)
            total_param += num_param
    return total_param


if __name__ == '__main__':
    import os
    # from torch.autograd import gradcheck, Variable

    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    device = 'cpu'
    model = HEKE_BVQA_resnet().to(device)
    print('number of trainable parameters = ', count_parameters(model))

    # inputs = Variable(torch.randn((1, 3, 8, 18, 18)), requires_grad=True).to(device)
    inputs = torch.randn((8, 3, 216, 384)).to(device)
    output = model(inputs)

    # test = gradcheck(lambda x: model(x), inputs, eps=1e-3)
    # print(test)
    print('done.')
