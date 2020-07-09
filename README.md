# pix2pix
pix2pix Project Group 1 Hakan, Delight, Dennis, Julien

# Documentation<br />
#### Week 1<br />
1. Created new Github Repository. 
2. Research pix2pix and read origincal paper.
3. Decide on Database (Cityscape). 

#### Week 2<br />
1. Created a template for our GAN (decided on Tensorflow for plotting purposes and trying something different)
2. Implemented subfunctions for randomizing of the data
3. Implemented the up- and downsampling functions

#### Week 3<br />
1. Implemented Generator/Discriminator
2. Implement training functions

## Run Pix2Pix GAN on Google Colab (The simple way)
Here is a Link where you can run our code on Google Colab. Just click on the link and press "run all". [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HakanDeymeci/vanila-gan/blob/master/Generative_Adversarial_Networks_PyTorch.ipynb)

## Overview of the Pix2Pix GAN
The Pix2Pix GAN is a Generative Adversarial Network. It performs image to image translation. The main parts are a Discriminator, a Generator and a dataset. In our case the dataset contains real images of cityscapes and sketches of cityscapes. The main goal of the GAN is that the Generator should produce fake images coming form the sketched images that can’t be differentiated with real images by the Discriminator. In other words: The Generator should make the Discriminator think that it is always getting real images even if they are generated. 
