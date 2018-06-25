import _init_path
from model import PartialUNet
from utils import to_var
import torchvision_sunner.transforms as sunnertransforms
import torchvision_sunner.data as sunnerData
import torchvision.transforms as transforms
import torch.nn as nn
import numpy as np
import argparse
import torch
import cv2
import os

"""
    This script can inpaint the image by using U-Net which adopt partial convolution technique
    You can select to assign the mask or not
"""

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default = './model.pth', type = str, help = 'The path of training model result')
    parser.add_argument('--image_path', default = "./data/test_input.jpg", type = str, help = 'The original image you want to deal with')
    parser.add_argument('--mask_path', default = None, type = str, help = 'The mask you want to adopt')
    parser.add_argument('--size', default = 256, type = int, help = 'The size of input')
    args = parser.parse_args()
    return args

def generateMask(img, mask_value = 0):
    """
        Generate the mask by specific value

        Arg:    img         - The numpy array object, rank format is HW3
                mask_value  - The value to represent the mask region
        Ret:    The mask array, rank format is HW3
    """
    h, w, c = np.shape(img)
    result = np.ones_like(img) * 255
    for i in range(h):
        for j in range(w):
            if img[i][j][0] == 0 and img[i][j][1] == 0 and img[i][j][2] == 0:
                result[i][j][0] = 0
                result[i][j][1] = 0
                result[i][j][2] = 0
    return result

if __name__ == '__main__':
    args = parse()

    # Load the model
    if not os.path.exists(args.model_path):
        raise Exception('You should train the model first...')
    model = PartialUNet()
    model = model.cuda() if torch.cuda.is_available() else model
    model.load_state_dict(torch.load(args.model_path))

    # Prepare image and mask
    img = cv2.imread(args.image_path)
    origin_size = (np.shape(img)[1], np.shape(img)[0])
    if args.mask_path is not None:
        mask = cv2.imread(args.mask_path)
    else:
        mask = generateMask(img)

    # Preprocessing
    proc_list = [
        sunnertransforms.Rescale((args.size, args.size)),
        sunnertransforms.ToTensor(),
        sunnertransforms.ToFloat(),
        sunnertransforms.Transpose(sunnertransforms.BHWC2BCHW),
        sunnertransforms.Normalize()
    ]
    for op in proc_list:
        img = op(img)
        mask = op(mask)
    img = torch.stack([img, img], 0)
    mask = torch.stack([mask, mask], 0)

    # Work!
    if args.mask_path is not None:
        model.setInput(target = img, mask = mask)
    else:
        model.setInput(image = img, mask = mask)        
    model.eval()
    model.forward()

    # Save
    _, recon_img, _ = model.getOutput()
    show_img = sunnertransforms.tensor2Numpy(recon_img, 
        transform = transforms.Compose([
            sunnertransforms.UnNormalize(),
            sunnertransforms.Transpose(sunnertransforms.BCHW2BHWC),
    ]))
    show_img = show_img.astype(np.uint8)
    show_img = cv2.resize(show_img[0], origin_size)
    cv2.imwrite("test.png", show_img)