from torch.autograd import Variable
import torch.nn as nn
import numpy as np
import torch

"""
    Implemention of the partial convolution layer
    The idea is come from NVIDIA paper: Image Inpainting for Irregular Holes Using Partial Convolutions
    Ref: https://arxiv.org/abs/1804.07723
    You should be careful that **this is not the official implementation**
    Also very thanks that the authors can give me some advise.

    To give respectation, you should site the original paper if you use the the concept of pconv. 
"""

class DifferentShapeException(Exception):
    """
        The exception to check if the shape of the input is invalid in partial convolution
    """
    def __init__(self, class_name, remind):
        print("[ %s ] >> " % class_name, remind)

class PartialConv2d(nn.Module):
    def __init__(self, input_channel = 1, output_channel = 32, kernel_size = 3, stride = 1, padding = 0):
        """
            The initialization of partial convolution. 

            Arg:    input_channel   - The number of the channel in the input tensor
                    output_channel  - The number of the channel in the output tensor
                    kernel_size     - The size of kernel in convolution process
                    stride          - The number of stride in convolution process
                    padding         - The number of zero padding
        """
        super(PartialConv2d, self).__init__()
        self.input_channel = input_channel
        self.output_channel = output_channel
        self.hadFixedKernel = False
        self.mask_conv = self.__Conv2d(
            input_channel = input_channel, 
            output_channel = output_channel, 
            kernel_size = kernel_size, 
            stride = stride, 
            padding = padding, 
            fill = 1, 
            bias = False
        )
        self.img_conv = self.__Conv2d(
            input_channel = input_channel, 
            output_channel = output_channel, 
            kernel_size = kernel_size, 
            stride = stride, 
            padding = padding,
            fill = None,
            bias = True
        )

    def __Conv2d(self, input_channel, output_channel, kernel_size, stride, padding, fill = None, bias = True):
        """
            Implement 2D convolution with zero padding.
            This function is the core part of partial convolution, 
            and it not recommand to call this function directly!

            This function would force to adopt zero padding additionally, 
            thus set the padding parameter as 0 in nn.Conv2d initialization

            Arg:    input_channel   - The number of the channel in the input tensor
                    output_channel  - The number of the channel in the output tensor
                    kernel_size     - The size of kernel in convolution process
                    stride          - The number of stride in convolution process
                    padding         - The number of zero padding
                    fill            - (default is None), the value that will be filled in the kernel
                                      that can compute the SUM(M)
                    bias            - (default is True), introduce the bias term or not
            Ret:    The nn.Sequential object which operate zero padding and convolution in order
        """
        op = []
        if padding != 0:
            op += [nn.ZeroPad2d(padding)]
        conv_op = nn.Conv2d(input_channel, output_channel, kernel_size, stride = stride, padding = 0, bias = bias)
        if fill is not None:
            conv_op.weight.data.fill_(1)
        op += [conv_op]
        return nn.Sequential(*op)

    def __fixedKernel(self):
        """
            Set the gradient revision as zero in mask convolution operation
            The mask is fixed that it don't need gradient computation
            This function will be called before the first forward of this layer
        """
        for v in self.mask_conv.parameters():
            v.requires_grad = False

    def forward(self, image, mask):
        """
            The forward process of partial convolution
            You should make sure that the size of image should be the same as the size of mask

            Arg:    image   - The image variable
                    mask    - The mask variable
            Ret:    The feature response map and revised mask
        """
        # Check if the shape of parameters is valid
        if image.size() != mask.size():
            raise DifferentShapeException(self.__class__.__name__,
                "The shape of 1st parameter should be the same as the shape of 2nd parameter!")
        if len(image.size()) != 4 or len(mask.size()) != 4:
            raise DifferentShapeException(self.__class__.__name__,
                "The shape of input paramemters should be obey (BCHW) format!")
        if image.size(1) != self.input_channel:
            raise DifferentShapeException(self.__class__.__name__,
                "The channel of image should be %d !" % self.input_channel)
        if mask.size(1) != self.input_channel:
            raise DifferentShapeException(self.__class__.__name__,
                "The channel of mask should be %d !" % self.input_channel)

        # Check if decline the mask kernel computation
        if not self.hadFixedKernel:
            self.__fixedKernel()
            self.hadFixedKernel = True

        # Construct the bias tensor first
        bias = self.img_conv(Variable(torch.zeros_like(image.data)))

        # Forward next
        mask = mask.float()
        sum_mask = self.mask_conv(mask)
        mask_image = (image * mask) 
        out = self.img_conv(mask_image)
        out = (out - bias) / sum_mask + bias
        mask = torch.clamp(sum_mask, 0, 1).int()
        return out, mask

if __name__ == '__main__':
    image = Variable(torch.from_numpy(np.random.random([32, 3, 320, 480])).float()).cuda()
    mask  = Variable(torch.from_numpy(np.random.randint(0, 2, [32, 3, 320, 480]))).cuda()
    net = PartialConv2d(3, 32)
    net.cuda()
    out, revised_mask = net(image, mask)