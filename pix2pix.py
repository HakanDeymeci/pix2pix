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

#Input datasets
dataset_for_training = tf.data.Dataset.list_files(PATH+'train/*.jpg')
dataset_for_training = dataset_for_training.map(load_image_train, num_parallel_calls=tf.data.experimental.AUTOTUNE)
dataset_for_training = dataset_for_training.shuffle(BUFFER_SIZE)
dataset_for_training = dataset_for_training.batch(BATCH_SIZE)

dataset_for_tests = tf.data.Dataset.list_files(PATH+'val/*.jpg')
dataset_for_tests = dataset_for_tests.map(load_image_test)
dataset_for_tests = dataset_for_tests.batch(BATCH_SIZE)


#Generator
def Generator():
  inputs = tf.keras.layers.Input(shape=[256,256,3])

  down_stack = get_down_stack()

  up_stack = get_up_stack()

  initializer = tf.random_normal_initializer(0., 0.02)
  last = tf.keras.layers.Conv2DTranspose(OUTPUT_CHANNELS, 4, strides=2, padding='same', kernel_initializer=initializer, activation='tanh') 

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

#Generator loss



#Discriminator

#Discriminator loss

#Optimizers


#Generating images


#Training the Gan
NUMBER_OF_EPOCHS = 300

def training(input_image, target, epoch):
  with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
    gen_output = generator(input_image, training=True)

    disc_real_output = discriminator([input_image, target], training=True)
    disc_generated_output = discriminator([input_image, gen_output], training=True)

    gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(disc_generated_output, gen_output, target)
    disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

  generator_gradients = gen_tape.gradient(gen_total_loss,
                                          generator.trainable_variables)
  discriminator_gradients = disc_tape.gradient(disc_loss,
                                               discriminator.trainable_variables)

  generator_optimizer.apply_gradients(zip(generator_gradients,
                                          generator.trainable_variables))
  discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
                                              discriminator.trainable_variables))

def training_loop(train_ds, epochs, test_ds):
  for epoch in range(epochs):

    display.clear_output(wait=True)
    print("Epoch: ", epoch)

    # Train
    for n, (input_image, target) in train_ds.enumerate():
      if n % 25 == 0:
        generate_images(generator, example_input, example_target)
      if (n+1) % 100 == 0:
        print()
      training(input_image, target, epoch)
    print()
    
training_loop(dataset_for_training, NUMBER_OF_EPOCHS, dataset_for_ests)
