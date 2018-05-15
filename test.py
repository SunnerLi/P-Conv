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

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default = './model.pth', type = str, help = 'The path of training model result')
    parser.add_argument('--image_path', default = "./data/test_input.jpg", type = str, help = 'The original image you want to deal with')
    parser.add_argument('--mask_path', default = "./data/mask/1.png", type = str, help = 'The mask you want to adopt')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse()

    # Load the model
    if not os.path.exists(args.model_path):
        raise Exception('You should train the model first...')
    model = PartialUNet()
    model = model.cuda() if torch.cuda.is_available() else model
    model.load_state_dict(torch.load(args.model_path))

    # Preprocessing
    proc_list = [
        sunnertransforms.Rescale((512, 512)),
        sunnertransforms.ToTensor(),
        sunnertransforms.ToFloat(),
        sunnertransforms.Transpose(sunnertransforms.BHWC2BCHW),
        sunnertransforms.Normalize()
    ]
    img = cv2.imread(args.image_path)
    mask = cv2.imread(args.mask_path)
    for op in proc_list:
        
        img = op(img)
        mask = op(mask)
    img = torch.unsqueeze(img, 0)
    mask = torch.unsqueeze(mask, 0)

    # Work and show the result
    model.setInput(target = img, mask = mask)
    model.forward()
    _, recon_img, _ = model.getOutput()
    show_img = sunnertransforms.tensor2Numpy(recon_img, 
        transform = transforms.Compose([
            sunnertransforms.UnNormalize(),
            sunnertransforms.Transpose(sunnertransforms.BCHW2BHWC),
    ]))
    show_img = show_img.astype(np.uint8)
    cv2.imwrite("test.png", show_img[0])