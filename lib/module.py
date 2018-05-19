from torch.autograd import Variable
from layers import PartialConv2d
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torch

"""
    Define the module that will be used to build the network
    The U-Net part is borrowed and modified from here:
    ( https://github.com/meetshah1995/pytorch-semseg/blob/master/ptsemseg/models/unet.py )
"""

class PartialDown(nn.Module):
    def __init__(self, input_channel = 3, output_channel = 32, kernel_size = 3, 
        stride = 2, padding = 1, bias = True, use_batch_norm = True, freeze = False):
        super(PartialDown, self).__init__()
        self.use_batch_norm = use_batch_norm
        if self.use_batch_norm:
            self.norm = nn.BatchNorm2d(output_channel)
            if freeze:
                for p in self.parameters():
                    p.requires_grad = False
        self.pconv = PartialConv2d(
            input_channel = input_channel,
            output_channel = output_channel,
            kernel_size = kernel_size,
            stride = stride,
            padding = padding
        )
        

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

class unetDown(nn.Module):
    def __init__(self, input_channel = 3, output_channel = 32, kernel_size = 3, 
        stride = 1, padding = 1, use_batch_norm = True, freeze = False):
        super(unetDown, self).__init__()           
        if use_batch_norm:
            batch = nn.BatchNorm2d(output_channel)
            if freeze:
                for p in batch.parameters():
                    p.requires_grad = False
            self.layer = nn.Sequential(
                nn.Conv2d(input_channel, output_channel, kernel_size, stride, padding),
                batch,
                nn.ReLU()
            )
        else:
            self.layer = nn.Sequential(
                nn.Conv2d(input_channel, output_channel, kernel_size, stride, padding),
                nn.ReLU()
            )

    def forward(self, inputs):
        return self.layer(inputs)

class unetUp(nn.Module):
    def __init__(self, input_channel = 32, concat_channel = 32, output_channel = 32, 
        kernel_size = 3, stride = 1, padding = 1, use_batch_norm = True, use_lr = True):
        super(unetUp, self).__init__()
        if use_batch_norm:
            self.layer = nn.Sequential(
                nn.Conv2d(input_channel + concat_channel, output_channel, kernel_size, stride, padding),
                nn.BatchNorm2d(output_channel),
                nn.ReLU()
            )
        else:
            self.layer = nn.Sequential(
                nn.Conv2d(input_channel + concat_channel, output_channel, kernel_size, stride, padding),
                nn.ReLU()
            )
        self.use_lr = use_lr
        self.up = nn.Upsample(scale_factor = 2, mode = 'nearest')

    def forward(self, inputs1, inputs2):
        outputs2 = self.up(inputs2)
        offset = outputs2.size()[2] - inputs1.size()[2]
        padding = 2 * [offset // 2, offset // 2]
        outputs1 = F.pad(inputs1, padding)
        return self.layer(torch.cat([outputs1, outputs2], 1))

if __name__ == '__main__':
    image = Variable(torch.from_numpy(np.random.random([32, 3, 160, 320])).float()).cuda()
    mask = Variable(torch.from_numpy(np.random.randint(0, 2, [32, 3, 160, 320]))).cuda()
    down = PartialDown(3, 64, 7, 2, 3, use_batch_norm = False).cuda()
    up = PartialUp(64, 3, 3, 3, 1, use_batch_norm = False).cuda()

    image_, mask_ = down(image, mask)
    image_, mask_ = up(image_, image, mask_, mask)
    print(image_.size())