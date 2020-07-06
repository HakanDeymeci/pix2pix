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

#Randomizing

#Generator

#Discriminator

#Optimizers

#train
