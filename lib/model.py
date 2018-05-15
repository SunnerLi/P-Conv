from module import PartialDown, PartialUp, unetConv2, unetUp
from torch.autograd import Variable
from CustomVGG import CustomVGG
from loss import GramL1Loss
from base import Model
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torch

class UNet(Model):
    def __init__(self, style_list = "p1,p2,p3", base = 64):
        super(UNet, self).__init__(style_list = style_list, base = base)

        # Define encoder layers
        self.down1 = unetConv2(3, base, False)
        self.pool1 = nn.MaxPool2d(kernel_size = 2)

        self.down2 = unetConv2(base * 1, base * 2)
        self.pool2 = nn.MaxPool2d(kernel_size = 2)

        self.down3 = unetConv2(base * 2, base * 4)
        self.pool3 = nn.MaxPool2d(kernel_size = 2)

        self.down4 = unetConv2(base * 4, base * 8)
        self.pool4 = nn.MaxPool2d(kernel_size = 2)

        self.down5 = unetConv2(base * 8, base * 8)
        self.pool5 = nn.MaxPool2d(kernel_size = 2)

        self.down6 = unetConv2(base * 8, base * 8)
        self.pool6 = nn.MaxPool2d(kernel_size = 2)

        self.down7 = unetConv2(base * 8, base * 8)
        self.pool7 = nn.MaxPool2d(kernel_size = 2)

        self.down8 = unetConv2(base * 8, base * 8)

        # Define decoder layers
        self.up8 = unetUp(512, 512, 512, False)
        self.up7 = unetUp(512, 512, 512, False)
        self.up6 = unetUp(512, 512, 512, False)
        self.up5 = unetUp(512, 512, 512, False)
        self.up4 = unetUp(512, 256, 256, False)
        self.up3 = unetUp(256, 128, 128, False)
        self.up2 = unetUp(128, 64, 64, False)
        self.up1 = unetConv2(64, 3, False)

        # Define dummy variable
        self.recon_mask = None        

    def forward(self):
        conv1 = self.down1(self.img)
        pool1 = self.pool1(conv1)

        conv2 = self.down2(pool1)
        pool2 = self.pool2(conv2)
        conv3 = self.down3(pool2)
        pool3 = self.pool3(conv3)
        conv4 = self.down4(pool3)
        pool4 = self.pool4(conv4)
        conv5 = self.down5(pool4)
        pool5 = self.pool5(conv5)
        conv6 = self.down6(pool5)
        pool6 = self.pool6(conv6)
        conv7 = self.down7(pool6)
        pool7 = self.pool7(conv7)
        center = self.down8(pool7)

        up8 = self.up8(conv7, center)
        up7 = self.up7(conv6, up8)
        up6 = self.up6(conv5, up7)
        up5 = self.up5(conv4, up6)
        up4 = self.up4(conv3, up5) 
        up3 = self.up3(conv2, up4)
        up2 = self.up2(conv1, up3)
        self.recon_img = self.up1(up2)
        self.recon_img = F.tanh(self.recon_img)

class PartialUNet(Model):
    def __init__(self, style_list = "p1,p2,p3", base = 64):
        super(PartialUNet, self).__init__(style_list = style_list, base = base)

        # Define encoder layers
        self.down1 = PartialDown(3, base, 7, 2, 3, use_batch_norm = False)
        self.down2 = PartialDown(base, base * 2, 5, 2, 2)
        self.down3 = PartialDown(base * 2, base * 4, 5, 2, 2)
        self.down4 = PartialDown(base * 4, base * 8, 3, 2, 1)
        self.down5 = PartialDown(base * 8, base * 8, 3, 2, 1)
        self.down6 = PartialDown(base * 8, base * 8, 3, 2, 1)
        self.down7 = PartialDown(base * 8, base * 8, 3, 2, 1)
        self.down8 = PartialDown(base * 8, base * 8, 3, 2, 1)

        # Define decoder layers
        self.up8 = PartialUp(512, 512, 512, 3, 1, 1)
        self.up7 = PartialUp(512, 512, 512, 3, 1, 1)
        self.up6 = PartialUp(512, 512, 512, 3, 1, 1)
        self.up5 = PartialUp(512, 512, 512, 3, 1, 1)
        self.up4 = PartialUp(512, 256, 256, 3, 1, 1)
        self.up3 = PartialUp(256, 128, 128, 3, 1, 1)
        self.up2 = PartialUp(128, 64, 64, 3, 1, 1)
        self.up1 = PartialUp(64, 3, 3, 3, 1, 1, use_batch_norm = False, use_lr = False)

    def forward(self):
        x1, m1 = self.down1(self.img, self.mask)
        x2, m2 = self.down2(x1, m1)
        x3, m3 = self.down3(x2, m2)
        x4, m4 = self.down4(x3, m3)
        x5, m5 = self.down5(x4, m4)
        x6, m6 = self.down6(x5, m5)
        x7, m7 = self.down7(x6, m6)
        x8, m8 = self.down8(x7, m7)

        x_, m_ = self.up8(x8, x7, m8, m7)
        x_, m_ = self.up7(x_, x6, m_, m6)
        x_, m_ = self.up6(x_, x5, m_, m5)
        x_, m_ = self.up5(x_, x4, m_, m4)
        x_, m_ = self.up4(x_, x3, m_, m3) 
        x_, m_ = self.up3(x_, x2, m_, m2)
        x_, m_ = self.up2(x_, x1, m_, m1)
        self.recon_img, self.recon_mask = self.up1(x_, self.img, m_, self.mask)
        self.recon_img = F.tanh(self.recon_img)

if __name__ == '__main__':
    net = UNet()
    img = Variable(torch.from_numpy(np.random.random([1, 3, 512, 512]))).float()
    mask = Variable(torch.from_numpy(np.random.randint(0, 1, [1, 3, 512, 512])))
    net.setInput(target = img, mask = mask)
    net = net.cuda()
    net()