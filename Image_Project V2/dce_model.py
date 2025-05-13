# dce_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class DCE_net(nn.Module):
    def __init__(self):
        super(DCE_net, self).__init__()

        self.relu = nn.ReLU(inplace=True)
        number_f = 32
        self.e_conv1 = nn.Conv2d(3, number_f, 3, 1, 1, bias=True)
        self.e_conv2 = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True)
        self.e_conv3 = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True)
        self.e_conv4 = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True)
        self.e_conv5 = nn.Conv2d(number_f * 2, number_f, 3, 1, 1, bias=True)
        self.e_conv6 = nn.Conv2d(number_f * 2, number_f, 3, 1, 1, bias=True)
        self.e_conv7 = nn.Conv2d(number_f * 2, 24, 3, 1, 1, bias=True)

    def forward(self, x):
        x1 = self.relu(self.e_conv1(x))
        x2 = self.relu(self.e_conv2(x1))
        x3 = self.relu(self.e_conv3(x2))
        x4 = self.relu(self.e_conv4(x3))
        x5 = self.relu(self.e_conv5(torch.cat([x3, x4], 1)))
        x6 = self.relu(self.e_conv6(torch.cat([x2, x5], 1)))
        x_r = torch.tanh(self.e_conv7(torch.cat([x1, x6], 1)))
        return self.enhance(x, x_r), x_r

    def enhance(self, x, A):
        # A: enhancement map with 8 R(x), each R(x) for one stage
        R = torch.split(A, 3, dim=1)
        x_enhance = x + R[0] * (torch.pow(x, 2) - x)
        for i in range(1, 8):
            x_enhance = x_enhance + R[i] * (torch.pow(x_enhance, 2) - x_enhance)
        return x_enhance

class ReverseDCEUNet(nn.Module):
    def __init__(self):
        super(ReverseDCEUNet, self).__init__()
        self.relu = nn.ReLU(inplace=True)

        def conv(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, 1, 1),
                nn.ReLU(inplace=True)
            )

        # Encoder
        self.e_conv1 = conv(3, 32)
        self.e_conv2 = conv(32, 64)
        self.e_conv3 = conv(64, 128)
        self.e_conv4 = conv(128, 256)

        # Decoder
        self.d_conv1 = conv(384, 128)
        self.d_conv2 = conv(192, 64)
        self.d_conv3 = conv(96, 32)
        self.d_conv4 = nn.Conv2d(35, 24, 3, 1, 1)  # 8 RGB * 3 = 24 eğri katsayısı

        self.pool = nn.MaxPool2d(2, 2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x):
        # Encoder
        x1 = self.e_conv1(x)
        x2 = self.e_conv2(self.pool(x1))
        x3 = self.e_conv3(self.pool(x2))
        x4 = self.e_conv4(self.pool(x3))

        # Decoder
        d1 = self.upsample(x4)
        d1 = self.d_conv1(torch.cat([d1, x3], 1))
        d2 = self.upsample(d1)
        d2 = self.d_conv2(torch.cat([d2, x2], 1))
        d3 = self.upsample(d2)
        d3 = self.d_conv3(torch.cat([d3, x1], 1))

        r = torch.tanh(self.d_conv4(torch.cat([d3, x], 1)))  # [-1, 1] arası

        x_enhanced = self.reverse_enhance(x, r)
        return x_enhanced, r

    def reverse_enhance(self, x, r):
        # DCE'nin tersi: parlaklığı azalt
        x_ = x
        for i in range(8):  # 8 iterasyon
            r_i = r[:, i*3:(i+1)*3, :, :]
            x_ = x_ - r_i * (x_ ** 2 - x_)
        return x_