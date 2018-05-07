from torch.autograd import Variable
from summary import summary
from model import PartialUNet
import numpy as np
import torch

if __name__ == '__main__':
    image = Variable(torch.from_numpy(np.random.random([1, 3, 512, 512])).float()).cuda()
    mask = Variable(torch.from_numpy(np.random.randint(0, 2, [1, 3, 512, 512]))).cuda()
    model = PartialUNet().cuda()
    summary(model, input_size = [(3, 512, 512), (3, 512, 512)])
    model(image, mask)