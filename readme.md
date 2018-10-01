# P-Conv
### The implementation of the partial convolution

[![Packagist](https://img.shields.io/badge/Pytorch-0.4.0+-red.svg)]()
[![Packagist](https://img.shields.io/badge/Torchvision-0.2.0-red.svg)]()
[![Packagist](https://img.shields.io/badge/Python-3.5.2-blue.svg)]()

![](https://github.com/SunnerLi/P-Conv/blob/master/data/readme_img/mix.png)

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
$ python3 train.py --epoch 10 --batch_size 4 --image_folder <Place365_dataset_folder> --mask_folder <mask_folder> --model_path pconv.pth --model_type pconv --record_time 100 --record_path pconv_1to10.csv --freeze False --lr 0.02 --lambda_style 1 

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

### Testing (New)
After 08/2018, we only provide the pre-trained model with size 256 which is trained again by ourself. You can refer to the original [page](https://github.com/SunnerLi/P-Conv/wiki/Readme.md) to download the rest pre-trained model. However, you can also perform the inpainting ability with this model right away! Just download from the following link.    

* The partial-convolution U-Net pre-trained result after 20 epoch on Place365 only, and the size of image is 256 x 256
```
https://drive.google.com/file/d/1Ss27dJ5TsQ39rCrUtEDnHt7eWst2jJyn/view?usp=sharing
```

After you download the model, you can check the inpainting result by this command. We provide the inpainting mechanism by using partial-convolution U-Net. If you want to try with traditional U-Net, just do a little revision in `test.py` slightly.     

```
python3 test.py --image_path data/test/kker_blur.jpg --model_path pconv_256_retrain.pth
```

Hardware & Time
---
The hardware we use is the server with GTX 1080Ti. For each model, we use single GPU to do the training. The total number of GPU we use is 3. The total training time is about 3 weeks. Moreover, we train again this summer in order to obtain the better performance while the training time is about 1 month.      

Result
---

![](https://github.com/SunnerLi/P-Conv/blob/master/data/readme_img/mix2.png)

The above image shows the render result which is sampled from original Place365. For simplicity, we only capture the image of air-plane. You can try to sample the other image by your own. In this figure, the lowest row is the image is captured by my friend, and it's not from official dataset. After we train again, the face of the person can be render normally, even though the perfect erasing is still not obtained.      

   
|  PSNR | Testing image  | Place365  |   
|---|---|---|
|  PConv | **28.73**  | **27.86**  |
|  U-Net | 13.37  | 18.62  |

We also provide the quantitative comparison. The above table shows the PSNR value toward different data distribution. The `Testing image` is the image with my friend, and the random 10 images is selected from Place365, and we name them as `Place365`. The mask is random selected which ratio of hole is between 0.01 to 0.2. The table shows that the model can really perform for the specific level of inpainting.    

Reference
---
[1] Guilin Liu, Fitsum A. Reda, Kevin J. Shih, Ting-Chun Wang, Andrew Tao, and Bryan Catanzaro, "Image Inpainting for Irregular Holes Using Partial Convolutions,"  arXiv: 1804.07723 [cs.CV], April 2018.
       