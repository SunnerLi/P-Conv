import _init_path
from skimage.measure import compare_psnr
from lib.model import PartialUNet, UNet
from tqdm import tqdm
import torchvision_sunner.transforms as sunnertransforms
import torchvision_sunner.data as sunnerData
import torchvision.transforms as transforms
import numpy as np
import argparse
import torch
import os

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default = './model.pth', type = str, help = 'The path of training model result')
    parser.add_argument('--model_type', default = 'pconv', type = str, help = 'The type of model (pconv or unet)')
    parser.add_argument('--folder_path', default = "./data/train/train2014", type = str, help = 'The folder of test image')
    parser.add_argument('--mask_path', default = "./data/train/mask", type = str, help = 'The mask you want to adopt')
    parser.add_argument('--size', default = 256, type = int, help = 'The size of input')
    args = parser.parse_args()
    return args

def evalModel(args, model):
    # Create data loader
    loader = sunnerData.ImageLoader(
        sunnerData.ImageDataset(root_list = [args.folder_path, args.mask_path], transform = transforms.Compose([
            sunnertransforms.Rescale((args.size, args.size)),
            sunnertransforms.ToTensor(),
            sunnertransforms.Transpose(sunnertransforms.BHWC2BCHW),
            sunnertransforms.Normalize()
        ]), sample_method = sunnerData.OVER_SAMPLING), 
        batch_size=1, 
        shuffle=False, 
        num_workers = 2
    )

    # Compute the PSNR and record
    psnr_list = []
    bar = tqdm(loader)
    for image, mask in bar:
        # Double the tensor to adapt with BN
        image = torch.cat([image, image], 0)
        mask = torch.cat([mask, mask], 0)

        # forward
        mask = (mask + 1) / 2
        model.setInput(target = image, mask = mask)
        model.forward()
        _, recon_img, _ = model.getOutput()
        psnr = compare_psnr(
            image[0].detach().cpu().numpy(), 
            recon_img[0].detach().cpu().numpy()
        )
        psnr_list.append(psnr)

    # Show the result
    print('\n\n')
    print('-' * 20, 'Complete evaluation', '-' * 20)
    print('Testing average psnr: %.4f' % np.mean(psnr_list))

if __name__ == '__main__':
    args = parse()

    # Load the trained model
    if args.model_type == 'pconv':
        if not os.path.exists(args.model_path):
            raise Exception('You should train the model first...')
        model = PartialUNet()
    if args.model_type == 'unet':
        if not os.path.exists(args.model_path):
            raise Exception('You should train the model first...')
        model = UNet()
    model = model.cuda() if torch.cuda.is_available() else model
    model.load_state_dict(torch.load(args.model_path))

    # Evaluate the PSNR
    evalModel(args, model)