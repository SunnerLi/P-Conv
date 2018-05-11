from torch.autograd import Variable
from layers import PartialConv2d
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torch

class PartialDown(nn.Module):
    def __init__(self, input_channel = 3, output_channel = 32, kernel_size = 3, 
        stride = 2, padding = 1, bias = True, use_batch_norm = True):
        super(PartialDown, self).__init__()
        self.use_batch_norm = use_batch_norm
        self.pconv = PartialConv2d(
            input_channel = input_channel,
            output_channel = output_channel,
            kernel_size = kernel_size,
            stride = stride,
            padding = padding
        )
        if self.use_batch_norm:
            self.norm = nn.BatchNorm2d(output_channel)

    def forward(self, x, m):
        x_, m_ = self.pconv(x, m)
        if self.use_batch_norm:
            self.norm(x_)
        x_ = F.relu(x_)
        return x_, m_

class PartialUp(nn.Module):
    def __init__(self, input_channel = 3, concat_channel = 64, output_channel = 32, 
        kernel_size = 3, stride = 2, padding = 1, bias = True, use_batch_norm = True, use_lr = True):
        super(PartialUp, self).__init__()
        self.use_batch_norm = use_batch_norm
        self.use_lr = use_lr
        self.pconv = PartialConv2d(
            input_channel = input_channel + concat_channel,
            output_channel = output_channel,
            kernel_size = kernel_size,
            stride = stride,
            padding = padding
        ) 
        self.up = nn.Upsample(scale_factor = 2, mode = 'nearest')
        if self.use_batch_norm:
            self.norm = nn.BatchNorm2d(output_channel)

    def checkAndPadding(self, var1, var2):
        if var1.size(2) > var2.size(2) or var1.size(3) > var2.size(3):
            var1 = var1[:, :, :var2.size(2), :var2.size(3)]
        else:
            pad = [0, 0, int(var2.size(2) - var1.size(2)), int(var2.size(3) - var1.size(3))]
            var1 = F.pad(var1, pad)
        return var1, var2

    def forward(self, x, cat_x, m, cat_m):
        x = self.up(x)
        m = self.up(m.float()).long()
        x, cat_x = self.checkAndPadding(x, cat_x)
        m, cat_m = self.checkAndPadding(m, cat_m)
        x = torch.cat([x, cat_x], 1)
        m = torch.cat([m, cat_m], 1)
        x_, m_ = self.pconv(x, m)
        if self.use_batch_norm:
            x_ = self.norm(x_)
        if self.use_lr:
            x_ = F.leaky_relu(x_, 0.2)
        return x_, m_

if __name__ == '__main__':
    image = Variable(torch.from_numpy(np.random.random([32, 3, 160, 320])).float()).cuda()
    mask = Variable(torch.from_numpy(np.random.randint(0, 2, [32, 3, 160, 320]))).cuda()
    down = PartialDown(3, 64, 7, 2, 3, use_batch_norm = False).cuda()
    up = PartialUp(64, 3, 3, 3, 1, use_batch_norm = False).cuda()

    image_, mask_ = down(image, mask)
    image_, mask_ = up(image_, image, mask_, mask)
    print(image_.size())