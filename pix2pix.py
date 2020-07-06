#libraries
import tensorflow as tf
import os
import time
from matplotlib import pyplot as plt
from IPython import display

#Import Dataset
DATA_URL = 'https://people.eecs.berkeley.edu/~tinghuiz/projects/pix2pix/datasets/cityscapes.tar.gz'

path_to_zip = tf.keras.utils.get_file('cityscapes.tar.gz',
                                      origin=DATA_URL,
                                      extract=True)

PATH = os.path.join(os.path.dirname(path_to_zip), 'cityscapes/')

#Loader
def load(image_file):
  image = tf.io.read_file(image_file)
  image = tf.image.decode_jpeg(image)

  diff = tf.shape(image)[1] // 2
  real = image[:, :diff, :]
  inp = image[:,diff:, :]

  inp = tf.cast(inp, tf.float32)
  real = tf.cast(real, tf.float32)

  return inp, real

#variables
BUFFER_SIZE = 400
BATCH_SIZE = 1
IMG_WIDTH = 256
IMG_HEIGHT = 256





#Randomizing



#crop and resize

def random_crop(inp, real):
  stacked_image = tf.stack([inp, real], axis=0)
  cropped_image = tf.image.random_crop(
      stacked_image, size=[2, IMG_HEIGHT, IMG_WIDTH, 3])

  return cropped_image[0], cropped_image[1]
#resize loaded data

def resize(inp, real, height, width):
  inp = tf.image.resize(inp, [height, width], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
  real = tf.image.resize(real, [height, width], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

  return inp, real


#normalize data
def normalize(inp, real):
  inp = (inp / 127.5) - 1
  real = (real / 127.5) - 1

  return inp, real

#mirroring
def mirroring(inp, real):
    inp = tf.image.flip_left_right(inp)
    real = tf.image.flip_left_right(real)

    return inp, real
  
  @tf.function()

def randomize(inp, real):
  if tf.random.uniform(()) > 0.5:
    inp, real = resize(inp, real, 286, 286)
    inp, real = random_crop(inp, real)
    inp, real = mirroring(inp, real)
  else:
    inp, real = resize(inp, real, 286, 286)
    inp, real = random_crop(inp, real)
    
    return inp, real
#sampling

#Generator

#Generator loss



#Discriminator

#Discriminator loss

#Optimizers


#Generating images


#Training the Gan
def fit(train_ds, epochs, test_ds):
  for epoch in range(epochs):
    start = time.time()

    display.clear_output(wait=True)
    print("Epoch: ", epoch)

    # Train
    for n, (input_image, target) in train_ds.enumerate():
      if n % 25 == 0:
        generate_images(generator, example_input, example_target)
      if (n+1) % 100 == 0:
        print()
      train_step(input_image, target, epoch)
    print()

    # saving (checkpoint) the model every 20 epochs
    if (epoch + 1) % 20 == 0:
      checkpoint.save(file_prefix = checkpoint_prefix)

    print ('Time taken for epoch {} is {} sec\n'.format(epoch + 1,
                                                        time.time()-start))
  checkpoint.save(file_prefix = checkpoint_prefix)
