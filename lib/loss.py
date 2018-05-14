import torch.nn.functional as F
import torch.nn as nn
import torch

# gram matrix and loss
class GramMatrix(nn.Module):
    def forward(self, input):
        b,c,h,w = input.size()
        F = input.view(b, c, h*w)
        G = torch.bmm(F, F.transpose(1,2)) 
        G.div_(h*w*c)
        return G

class GramL1Loss(nn.Module):
    def forward(self, input, target):
        out = nn.L1Loss()(GramMatrix()(input), GramMatrix()(target))
        return(out)