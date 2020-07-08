import tensorflow as tf
import os
import time
from matplotlib import pyplot as plt
from IPython import display

DATA_URL = 'https://people.eecs.berkeley.edu/~tinghuiz/projects/pix2pix/datasets/cityscapes.tar.gz'

path_to_zip = tf.keras.utils.get_file('cityscapes.tar.gz',
                                      origin=DATA_URL,
                                      extract=True)

PATH = os.path.join(os.path.dirname(path_to_zip), 'cityscapes/')

BUFFER_SIZE = 400
BATCH_SIZE = 1
IMG_WIDTH = 256
IMG_HEIGHT = 256

def load(image_file):
  image = tf.io.read_file(image_file)
  image = tf.image.decode_jpeg(image)

  diff = tf.shape(image)[1] // 2
  real = image[:, :diff, :]
  inp = image[:,diff:, :]

  inp = tf.cast(inp, tf.float32)
  real = tf.cast(real, tf.float32)

  return inp, real

inp, re = load(PATH+'train/100.jpg')

def resize(inp, real, height, width):
  inp = tf.image.resize(inp, [height, width], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
  real = tf.image.resize(real, [height, width], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

  return inp, real

def random_crop(inp, real):
  stacked_image = tf.stack([inp, real], axis=0)
  cropped_image = tf.image.random_crop(
      stacked_image, size=[2, IMG_HEIGHT, IMG_WIDTH, 3])

  return cropped_image[0], cropped_image[1]

def mirroring(inp, real):
    inp = tf.image.flip_left_right(inp)
    real = tf.image.flip_left_right(real)

    return inp, real
  
def normalize(inp, real):
  inp = (inp / 127.5) - 1
  real = (real / 127.5) - 1

  return inp, real

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
  
def load_image_train(image_file):
  input_image, real_image = load(image_file)
  input_image, real_image = randomize(input_image, real_image)
  input_image, real_image = normalize(input_image, real_image)

  return input_image, real_image

def load_image_test(image_file):
  input_image, real_image = load(image_file)
  input_image, real_image = resize(input_image, real_image,
                                   IMG_HEIGHT, IMG_WIDTH)
  input_image, real_image = normalize(input_image, real_image)

  return input_image, real_image

dataset_for_training = tf.data.Dataset.list_files(PATH+'train/*.jpg')
dataset_for_training = dataset_for_training.map(load_image_train, num_parallel_calls=tf.data.experimental.AUTOTUNE)
dataset_for_training = dataset_for_training.shuffle(BUFFER_SIZE)
dataset_for_training = dataset_for_training.batch(BATCH_SIZE)

dataset_for_tests = tf.data.Dataset.list_files(PATH+'val/*.jpg')
dataset_for_tests = dataset_for_tests.map(load_image_test)
dataset_for_tests = dataset_for_tests.batch(BATCH_SIZE)

OUTPUT_CHANNELS = 3
bias = False

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
  
def Generator():
  inputs = tf.keras.layers.Input(shape=[256,256,3])

  down_stack = get_down_stack()

  up_stack = get_up_stack()

  initializer = tf.random_normal_initializer(0., 0.02)
  last = tf.keras.layers.Conv2DTranspose(OUTPUT_CHANNELS, 4,
                                         strides=2,
                                         padding='same',
                                         kernel_initializer=initializer,
                                         activation='tanh') 

  all_inputs = inputs

  skips = []
  for down in down_stack:
    all_inputs = down(all_inputs)
    skips.append(all_inputs)

  skips = reversed(skips[:-1])

  for up, skip in zip(up_stack, skips):
    all_inputs = up(all_inputs)
    all_inputs = tf.keras.layers.Concatenate()([all_inputs, skip])

  all_inputs = last(all_inputs)

  return tf.keras.Model(inputs=inputs, outputs=all_inputs)

def get_down_stack():
  down_stack = [
    downsample(64, 4, apply_batchnorm=False), 
    downsample(128, 4), 
    downsample(256, 4), 
    downsample(512, 4), 
    downsample(512, 4), 
    downsample(512, 4), 
    downsample(512, 4), 
    downsample(512, 4), 
  ]
  return down_stack

def get_up_stack():
  up_stack = [
    upsample(512, 4, apply_dropout=True), 
    upsample(512, 4, apply_dropout=True), 
    upsample(512, 4, apply_dropout=True), 
    upsample(512, 4), 
    upsample(256, 4), 
    upsample(128, 4), 
    upsample(64, 4), 
  ]
  return up_stack

generator = Generator()
