# CartoonGAN-tensorflow
Simple code implement the paper of CartoonGAN

## Introduction
This simple code mainly implement the paper about [CartoonGAN](http://openaccess.thecvf.com/content_cvpr_2018/papers/Chen_CartoonGAN_Generative_Adversarial_CVPR_2018_paper.pdf), which is published at CVPR2018. This paper mainly address the problem of styling the nature images to cartoon by using GANs.

#### There are three contributions in the paper:
1. Removing the clear edge of original cartoon images as a new datasets
2. Using high-level feature maps in vgg19 to reconstruct the content
3. Initializing the generator by reconstructing the nature images

## Results of the paper
![](https://github.com/MingtaoGuo/CartoonGAN-tensorflow/blob/master/images/paperresult.jpg)

## How to use the code
#### Python packages you need:
1. python 3.x
2. tensorflow 1.4.0
3. pillow
4. scipy
5. cv2
6. numpy
#### Process of using
1. Download the cartoon movie, then using 'vedio2img.py' to extract the cartoon images from the cartoon movie. Finally, put the extracted cartoon imges into the folder 'c'.
2. Using 'remove_clear_edge.py' to remove the extracted cartoon images' clear edge as a new datasets, and then put the unclear edge cartoon images into the folder 'e'
3. Download the nature image datasets(we use [MSCOCO]() here, not Flicker)
## Is training ..........
---------------------
