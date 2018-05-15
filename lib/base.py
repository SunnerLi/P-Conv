from torch.autograd import Variable
from CustomVGG import CustomVGG
from loss import GramL1Loss
from utils import to_var
import torch.nn as nn
import torch

class Model(nn.Module):
    def __init__(self, style_list = "p1,p2,p3", base = 64):
        super(Model, self).__init__()
        self.style_list = style_list.split(',')

        # Set loss balance constants
        self.lambda_hole = 6
        self.lambda_perceptual = 0.05
        self.lambda_style = 1
        self.lambda_tv = 0.1

        # Load pre-trained VGG16 and fix parameter training
        self.vgg = CustomVGG(model_path = 'vgg_conv.pth', download = True)
        for v in self.vgg.parameters():
            v.requires_grad = False

        # Define criterion
        self.criterion_l1    = nn.L1Loss()
        self.criterion_style = GramL1Loss()

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

    def getTrainableParameters(self):
        trainable_list = [v for v in self.parameters() if v.requires_grad is True]
        return (v for v in trainable_list)

    def getOutput(self):
        return self.img, self.recon_img, self.recon_mask

    def forward(self):
        pass

    def backward(self):
        # Formulate the symbol
        I_out  = self.recon_img
        I_gt   = self.target
        M      = self.mask.clone().float()
        I_comp = M * I_out + (1 - M) * I_gt

        # Pass through VGG
        psi_out  = self.vgg(I_out, self.style_list)
        psi_gt   = self.vgg(I_gt, self.style_list)
        psi_comp = self.vgg(I_comp, self.style_list)

        # ------------------------------------------------------------------------------------------------
        # Form loss term
        # ------------------------------------------------------------------------------------------------
        loss_valid = self.criterion_l1(M * I_out, M * I_gt)
        loss_hole = self.criterion_l1((1 - M) * I_out, (1 - M) * I_gt)
        loss_perceptual = None
        loss_style_out = None
        loss_style_comp = None
        for layer in self.style_list:
            # Perceptual loss
            if loss_perceptual is None:
                loss_perceptual = self.criterion_l1(psi_out[layer], psi_gt[layer])
            else:
                loss_perceptual += self.criterion_l1(psi_out[layer], psi_gt[layer])

            # Style loss (raw output)
            if loss_style_out is None:
                loss_style_out = self.criterion_style(psi_out[layer], psi_gt[layer])
            else:
                loss_style_out += self.criterion_style(psi_out[layer], psi_gt[layer])

            # Style loss (composited output)
            if loss_style_comp is None:
                loss_style_comp = self.criterion_style(psi_comp[layer], psi_gt[layer])
            else:
                loss_style_comp += self.criterion_style(psi_comp[layer], psi_gt[layer])

        # TV loss
        b, c, h, w = I_comp.size()
        vertical_target = Variable(I_comp[:, :, 1:, :].data, requires_grad = False)
        horizontal_target = Variable(I_comp[:, :, :, 1:].data, requires_grad = False)
        loss_tv = self.criterion_l1(I_comp[:, :, :h-1, :], vertical_target) + \
            self.criterion_l1(I_comp[:, :, :, :w-1], horizontal_target)

        # Merge as total loss
        loss_total = loss_valid \
            + self.lambda_hole * loss_hole \
            + self.lambda_perceptual * loss_perceptual \
            + self.lambda_tv * loss_tv \
            + self.lambda_style * (loss_style_out + loss_style_comp) 
        loss_total.backward()
        return loss_total