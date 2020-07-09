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

# Explanation of our Code 
We start with importing the dependencies we need for our GAN.
## Imports
```
import tensorflow as tf
import os
import time
from matplotlib import pyplot as plt
from IPython import display
```
## Dataset
We are using a dataset that contains images of real cityscapes and images of sketched cityscapes that are sized 256 x 256.
```
DATA_URL = 'https://people.eecs.berkeley.edu/~tinghuiz/projects/pix2pix/datasets/cityscapes.tar.gz'

path_to_zip = tf.keras.utils.get_file('cityscapes.tar.gz',
                                      origin=DATA_URL,
                                      extract=True)

PATH = os.path.join(os.path.dirname(path_to_zip), 'cityscapes/')
```
```
BUFFER_SIZE = 400
BATCH_SIZE = 1
IMG_WIDTH = 256
IMG_HEIGHT = 25
```
## Loader
Next the load function loads and decodes the images of the dataset we defined before.
```
def load(image_file):
  image = tf.io.read_file(image_file)
  image = tf.image.decode_jpeg(image)

  diff = tf.shape(image)[1] // 2
  real = image[:, :diff, :]
  inp = image[:,diff:, :]

  inp = tf.cast(inp, tf.float32)
  real = tf.cast(real, tf.float32)

  return inp, real
```
```
inp, re = load(PATH+'train/100.jpg')
```
## Randomizing
The randomizing section contains three functions that are getting applied by the randomize() function.
### Resize
Here the input and real images can be resizes with a given height and width.
```
def resize(inp, real, height, width):
  inp = tf.image.resize(inp, [height, width], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
  real = tf.image.resize(real, [height, width], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

  return inp, real
```
### Random Crop
In this function the images are randomly cropped. 
```
def random_crop(inp, real):
  stacked_image = tf.stack([inp, real], axis=0)
  cropped_image = tf.image.random_crop(
      stacked_image, size=[2, IMG_HEIGHT, IMG_WIDTH, 3])

  return cropped_image[0], cropped_image[1]

```
### Mirroring
Next up the images are getting flipped horizontaly.
```
def mirroring(inp, real):
    inp = tf.image.flip_left_right(inp)
    real = tf.image.flip_left_right(real)

    return inp, real
```
### Randomize
The randomize function applies the three described function form before.
```
@tf.function()
def randomize(inp, real):
  if tf.random.uniform(()) > 0.5:
    inp, real = resize(inp, real, 286, 286)
    inp, real = random_crop(inp, real)
    inp, real = mirroring(inp, real)

    return inp, real
  else:
    inp, real = resize(inp, real, 286, 286)
    inp, real = random_crop(inp, real)
    
    return inp, real
```
## Normalize
The values of the dataset images are between 0 and 255 so we normailzed them to numbers between -1 and 1.
```
def normalize(inp, real):
  inp = (inp / 127.5) - 1
  real = (real / 127.5) - 1

  return inp, real
```
## Load train and test data
Moving on we loaded the images of our dataset.
```
def load_image_train(image_file):
  input_image, real_image = load(image_file)
  input_image, real_image = randomize(input_image, real_image)
  input_image, real_image = normalize(input_image, real_image)

  return input_image, real_image
```
```
def load_image_test(image_file):
  input_image, real_image = load(image_file)
  input_image, real_image = resize(input_image, real_image,
                                   IMG_HEIGHT, IMG_WIDTH)
  input_image, real_image = normalize(input_image, real_image)

  return input_image, real_image
```
```
dataset_for_training = tf.data.Dataset.list_files(PATH+'train/*.jpg')
dataset_for_training = dataset_for_training.map(load_image_train,
                                  num_parallel_calls=tf.data.experimental.AUTOTUNE)
dataset_for_training = dataset_for_training.shuffle(BUFFER_SIZE)
dataset_for_training = dataset_for_training.batch(BATCH_SIZE)

```
```
dataset_for_tests = tf.data.Dataset.list_files(PATH+'val/*.jpg')
dataset_for_tests = dataset_for_tests.map(load_image_test)
dataset_for_tests = dataset_for_tests.batch(BATCH_SIZE)
```
```
OUTPUT_CHANNELS = 3
bias = False
```
## Sampling
```
def downsample(filters, size, apply_batchnorm=True):

  result = tf.keras.Sequential()
  result.add(
      tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                             kernel_initializer=tf.random_normal_initializer(0., 0.03), use_bias=False))
  if apply_batchnorm:
    result.add(tf.keras.layers.BatchNormalization())
    result.add(tf.keras.layers.LeakyReLU())
    return result
  else:
    result.add(tf.keras.layers.LeakyReLU())
  return result
```
```
def upsample(filters, size, apply_dropout=False):
  result = tf.keras.Sequential()
  result.add(
    tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                    padding='same',kernel_initializer=tf.random_normal_initializer(0., 0.03),use_bias=False))
  result.add(tf.keras.layers.BatchNormalization())

  if apply_dropout:
      result.add(tf.keras.layers.Dropout(0.5))
      result.add(tf.keras.layers.ReLU())
      return result
  else:
    result.add(tf.keras.layers.ReLU())
    return result
```
