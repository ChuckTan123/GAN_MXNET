# GAN_MXNET

This tutorial is the implementation of animeGAN in MXNET from https://github.com/jayleicn/animeGAN

To run the code:

    1.Download the rec data from google drive
        https://drive.google.com/drive/u/0/folders/1cl-lmAaKgN2Y6GoYER8J34BlbS4cDviM
        
        The detail code to generate .rec can be found in ./data/
    
    2. run the command under the root "python run_train"
    
        The generated images at epoch 124 are like:

![fake_sample_1](epoch124.png)

Useful links:
    
https://github.com/jayleicn/animeGAN

https://github.com/apache/incubator-mxnet/tree/master/example/gan
    
https://github.com/ChuckTan123/ganhacks
    
https://github.com/carpedm20/DCGAN-tensorflow

https://arxiv.org/abs/1511.06434

Learn GAN:
The review paper in 2017: 

https://arxiv.org/pdf/1711.05914.pdf

Notes:

GAN does not need to do further distribution assumption and can simply infer real-like samples
from latent space. This powerful property leads GAN to be applied various applications such
as image synthesis, image attribute editing, image translation, domain adaptation and even
to other academic fields. 

Why Generative Models: 

By training generative models, it can produce real-like
data which can be used in filling missing data in semi-supervised learning

Also, it can change image to other specific domain in supervised manner or unsupervised manner and even guide image
to be transformed to possess specific attributes we want.

Fill the missing data point for semi-supervise learning. 

GAN doesn't need to explicit the distribution of the data. 

First of all, we represent G and D as deep neural networks to learn parameters rather than
directly learning pθ(x) itself. Modeling with deep neural networks such as MLP is advantageous
in that parameters of distributions can be easily learned through gradient descent using backpropagation
and it does not need further distribution assumptions to do an inference, rather it can
generate samples following pθ(x) through a simple feed-forward. But this practical implementation
derives a gap with theory.

Applications using GAN: 

    1.Image Translation:
        Image translation is translating images from domain X to images from other domain Y. Mainly,
        translated images have dominants characteristic of domain Y maintaining attributes of before
        translated. As in classical machine learning, there are supervised and unsupervised ways to conduct
        image translation.
