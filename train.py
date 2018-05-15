import _init_path
from torch.autograd import Variable
from model import PartialUNet, UNet
from torch.optim import Adam
from summary import summary
import torchvision_sunner.transforms as sunnertransforms
import torchvision_sunner.data as sunnerData
import torchvision.transforms as transforms
import torch.nn as nn
import pandas as pd
import numpy as np
import argparse
import torch
import cv2
import os

def parse():
    parser = argparse.ArgumentParser()

    # Fundemental
    parser.add_argument('--epoch', default = 1, type = int, help = 'epoch')
    parser.add_argument('--batch_size', default = 1, type = int, help = 'batch size')
    parser.add_argument('--image_folder', default = './data/train2014', type = str, help = 'The folder of the training image')
    parser.add_argument('--model_path', default = './model.pth', type = str, help = 'The path of training model result')
    parser.add_argument('--model_type', default = 'pconv', type = str, help = 'The type of model (pconv or unet)')   
    parser.add_argument('--record_time', default = 100, type = int, help = 'The period to record the training result')   
    parser.add_argument('--record_path', default = 'pconv.csv', type = str, help = 'The CSV name of record file')   
    parser.add_argument('--style', default = "p1,p2,p3", type = str, help = 'The symbol of style loss')

    # Hyper-parameters
    parser.add_argument('--lr', default = 0.0002, type = float, help = 'The learning rate')   
    parser.add_argument('--lambda_style', default = 1, type = float, help = 'The weight of style loss')   

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse()

    # Create data loader
    sunnerData.quiet()
    sunnertransforms.quiet()
    loader = sunnerData.ImageLoader(
        sunnerData.ImageDataset(root_list = [args.image_folder, './data/mask'], transform = transforms.Compose([
            # sunnertransforms.Rescale((360, 640)),
            sunnertransforms.Rescale((512, 512)),
            sunnertransforms.ToTensor(),

            # BHWC -> BCHW
            sunnertransforms.Transpose(sunnertransforms.BHWC2BCHW),
            sunnertransforms.Normalize()
        ])), 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers = 2
    )

    # Load model
    if args.model_type == 'pconv':
        model = PartialUNet(style_list = args.style, base = 64, style_weight = args.lambda_style)
    elif args.model_type == 'unet':
        model = UNet(style_list = args.style ,base = 64, style_weight = args.lambda_style)
    else:
        raise Exception('Model type is not support... (Just accept pconv or unet)')    
    model = model.cuda() if torch.cuda.is_available() else model
    if os.path.exists(args.model_path):
        model.load_state_dict(torch.load(args.model_path))
    optimizer = Adam(model.getTrainableParameters(), lr = args.lr)

    # Train
    loss_list = []
    for epoch in range(args.epoch):
        for idx, (image, mask) in enumerate(loader):
            
            # forward
            mask = (mask + 1) / 2
            model.setInput(target = image, mask = mask)
            model.forward()

            # backward
            optimizer.zero_grad()
            loss = model.backward()
            optimizer.step()

            # Show
            if idx % args.record_time == 0:
                print('Epoch: ', epoch, '\tIteration: ', idx, '\tLoss: ', loss.data[0])
                input_img, recon_img, recon_mask = model.getOutput()
                show_img = sunnertransforms.tensor2Numpy(recon_img, 
                    transform = transforms.Compose([
                        sunnertransforms.UnNormalize(),
                        sunnertransforms.Transpose(sunnertransforms.BCHW2BHWC),
                ]))
                show_img = show_img.astype(np.uint8)
                cv2.imwrite("show.png", show_img[0])
                torch.save(model.state_dict(), args.model_path)

                loss_list.append(loss.data[0])
                df = pd.DataFrame({'loss': loss_list})
                df.to_csv(args.record_path)