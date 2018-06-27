# P-Conv
### The implementation of the partial convolution

[![Packagist](https://img.shields.io/badge/Pytorch-0.4.0+-red.svg)]()
[![Packagist](https://img.shields.io/badge/Torchvision-0.2.0-red.svg)]()
[![Packagist](https://img.shields.io/badge/Python-3.5.2-blue.svg)]()

![](https://cdn-images-1.medium.com/max/2000/1*HUmj7An3CvGrJiTZAgiHBw.png)


Abstraction
---
This project is trying to re-produce the concept of the Nvidia's paper - [Image Inpainting for Irregular Holes Using Partial Convolutions](https://arxiv.org/abs/1804.07723)[1]. **You should notice that it's not the official implementation. Besides, this project doesn't re-produce the full result in the paper!** The next section will introduce the detail of this project.    

Introduction
---
**You should notice the difference of the dataset!!** In this repo, we try to use **Places365-Standard (not ImageNet)** to train the model from scratch since we don't have extra secondary storage to store whole ImageNet training data, and we do the pre-training for 20 epochs. The Place365-Standard dataset contains 1.8 millions images. Second, we use **MS-COCO 2014 training dataset** to fine tune the model. The number of epoch in MS-COCO 2014 training is 400.    

According to the concept of partial convolution, it's no influence to do the zero padding toward both image and mask. By this idea, we try to fine tune and measure the performance with size of 224 * 224. We also provide the traditional U-Net to evaluate the performance at the same time.        

For the last ,another points you should notice is that it's hard to re-produce the perfect result by using the hyper-parameters which paper had mentioned. Thus, we adopt two-stage training idea. In the first stage, the weight of style loss is set to 1 for 10 epoch. Next, the weight of style loss will be recovered as 120 for another 10 epoch which might avoid Fish Scale Artifacts.     

Usage
---
### Training
If you want to train from scratch, you can choose to train directly, or train with two-stage strategy. The following list the command you should type.    
<br/>
Train partial-convolution U-Net: 
```
# Pre-train (1st stage)
$ python3 train.py --epoch 10 --batch_size 4 --image_folder <Place365_dataset_folder> --model_path pconv.pth --model_type pconv --record_time 100 --record_path pconv_1to10.csv --freeze False --lr 0.02 --lambda_style 1 

# Pre-train (2nd stage)
$ python3 train.py --epoch 10 --batch_size 4 --image_folder <Place365_dataset_folder> --model_path pconv.pth --model_type pconv --record_time 100 --record_path pconv_11to20.csv --freeze False --lr 0.002 --lambda_style 120 

# Remain the pre-trained result before you fine-tune
$ cp pconv.pth pconv_tune.pth

# Fine-tune
$ python3 train.py --epoch 400 --batch_size 4 --image_folder <MSCOCO_dataset_folder> --model_path pconv_tune.pth --model_type pconv --record_time 100 --record_path pconv_tune.csv --freeze True --lr 0.0002 --lambda_style 120 
```

Train traditional U-Net:
```
# Pre-train (1st stage)
$ python3 train.py --epoch 10 --batch_size 4 --image_folder <Place365_dataset_folder> --model_path unet.pth --model_type unet --record_time 100 --record_path unet_1to10.csv --freeze False --lr 0.02 --lambda_style 1 

# Pre-train (2nd stage)
$ python3 train.py --epoch 10 --batch_size 4 --image_folder <Place365_dataset_folder> --model_path unet.pth --model_type unet --record_time 100 --record_path unet_11to20.csv --freeze False --lr 0.002 --lambda_style 120 

# Remain the pre-trained result before you fine-tune
$ cp unet.pth unet_tune.pth

# Fine-tune
$ python3 train.py --epoch 400 --batch_size 4 --image_folder <MSCOCO_dataset_folder> --model_path unet_tune.pth --model_type unet --record_time 100 --record_path unet_tune.csv --freeze True --lr 0.0002 --lambda_style 120 
```

### Testing
We also provide our training result to let you perform the inpainting ability right away! Just download from the following link.    

1. The partial-convolution U-Net pre-trained result after 20 epoch on Place365 only, and the size of image is 256 x 256
```
https://drive.google.com/file/d/1_r0Ek2CaPGfiIcchpxdzBbUrELjZwRNN/view?usp=sharing
```

2. The partial-convolution U-Net result after fine tune on MS-COCO 2014, and the size of image is 256 x 256
```
https://drive.google.com/file/d/1noLXmAkCiSn7-LLPS8lpy4jVc_Br0paa/view?usp=sharing
```

3. The traditional U-Net pre-trained result after 20 epoch on Place365 only, and the size of image is 256 x 256
```
https://drive.google.com/file/d/1YLGXLndVgM0CkRd_s8-lgaGV_QUwG7rx/view?usp=sharing
```

4. The partial-convolution U-Net pre-trained result after 20 epoch on Place365 only, and the size of image is 224 x 224
```
https://drive.google.com/file/d/1VUxNQQXIRP6n92cT-z2F6rtDIOwxEyoo/view?usp=sharing
```
<br/>    

After you download the model, you can check the inpainting result by this command. We provide the inpainting mechanism by using partial-convolution U-Net. If you want to try with traditional U-Net, just do a little revision in `test.py` slightly.     

<br/>     
    
```
python3 test.py --image_path data/test/kker_blur.jpg --model_path pconv3_fine_tune_coco.pth
```

Hardware & Time
---
The hardware we use is the server with GTX 1080Ti. For each model, we use single GPU to do the training. The total number of GPU we use is 3. The total training time is about 3 weeks. (2 weeks pre-train and 1 week fine-tune)  

Result
---
![](https://raw.githubusercontent.com/SunnerLi/partial/master/appendix/merge.png?token=AK99R3xvYG86Nr19bZvw6tUD2rHB0AjZks5bPJJmwA%3D%3D)
The training loss curve is shown above. We record the loss for each 100 iteration, and list the front 20 epoch. As you can see, the convergence for the last 10 epoch is still instable. However, the trend of size 256 is very similar to the size 224. It shows that the partial convolution U-Net can indeed adapt for arbitrary size (not the exponential of 2).         

![](https://raw.githubusercontent.com/SunnerLi/partial/master/appendix/kker_merge.png?token=AK99RwZTKaWznAwZhqnVMjkL6aalOE1mks5bPM3WwA%3D%3D)

The figure illustrates the render result. This image is captured by my friend, and it's not from official dataset.  However, the perfect result which paper showed cannot be seen in this re-implementation. The face region in the image is just like rendering with colorful style. On the other hand, the U-Net tends to render with identity mapping.    

During training, the convergence of U-Net is much faster than partial U-Net. Some level of convergence in partial U-Net can also be seen in training stage. We guess that the partial convolution can only learn the meaningful feature after **enough tremendous of training**. The performance is limit if the training is not enough.    

|   | Partial U-Net  | U-Net  |   
|---|---|---|
| PSNR  | 22.65  | 13.37  |

The above table shows the PSNR value toward the testing image. Even though the Partial U-Net can get higher score, the result is not satisfied. You can try the following command to evaluate the other dataset by yourself.    

```
# Evaluate the performance of partial U-Net
$ python3 eval.py --model_type pconv --model_path pconv3_fine_tune_mscoco.pth --folder_path <image_folder> --mask_path  <mask_folder>

# Evaluate the performance of traditional U-Net
$ python3 eval.py --model_type unet --model_path unet_256.pth --folder_path <image_folder> --mask_path <mask_folder>
```

Reference
---
[1] Guilin Liu, Fitsum A. Reda, Kevin J. Shih, Ting-Chun Wang, Andrew Tao, and Bryan Catanzaro, "Image Inpainting for Irregular Holes Using Partial Convolutions,"  arXiv: 1804.07723 [cs.CV], April 2018.
       