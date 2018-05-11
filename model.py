from module import PartialDown, PartialUp
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torch

def to_var(obj):
    if obj is not None:
        return Variable(obj) if type(obj) != Variable else obj
    return None

class PartialUNet(nn.Module):
    def __init__(self, base = 64):
        super(PartialUNet, self).__init__()

        # Set loss balance constants
        self.lambda_hole = 6
        self.lambda_perceptual = 0.05

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

        # Define criterion
        self.criterion = nn.MSELoss()

    def setInput(self, image = None, mask = None, target = None):
        # Judge if the answer is gotten
        self.target = to_var(target)
        self.mask = to_var(mask)
        if image is None:
            self.img = self.target.clone() * self.mask.clone().float()
        else:
            self.img = to_var(image)
        self.mask = self.mask.long()

        # Move to gpu if gpu is availiable
        if torch.cuda.is_available():
            self.img = self.img.cuda()
            self.mask = self.mask.cuda()
            if self.target is not None:
                self.target = self.target.cuda()

    def getOutput(self):
        return self.recon_img, self.recon_mask

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

    def backward(self):
        loss = self.criterion(self.recon_img, self.img)
        loss.backward()
        return loss