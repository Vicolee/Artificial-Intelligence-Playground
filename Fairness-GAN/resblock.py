# To create ResNet blocks and use Batch Normalization. Using "Improved Training with Wasserstein GAN model"
from spectral_normalization import DenseSN, ConvSN2D
import tensorflow as tf
import numpy as np
from keras import layers
from keras.utils import plot_model
from keras.models import Model
import keras.backend as K

# By default, doubles the output height and width due to upsampling layer.
def G_ResBlock(X, stage):
    # To input one that has been reshaped
    # Shortcut layers
    if stage == 1:
        num_filters = 1024
    elif stage == 2:
        num_filters = 512
    elif stage == 3:
        num_filters = 256
    elif stage == 4:
        num_filters = 128
    else:
        num_filters = 1024
    X_shortcut = layers.UpSampling2D((2,2), data_format='channels_last', name='G_ResBlock_Shortcut'+str(stage))(X)
    X_shortcut = layers.Conv2D(num_filters, kernel_size=3, strides=1, kernel_initializer='glorot_uniform', bias_initializer='zeros', padding='same')(X_shortcut)
    X_shortcut = layers.BatchNormalization()(X_shortcut)
    X_shortcut = layers.ReLU()(X_shortcut)
    X_shortcut = layers.Conv2D(num_filters, kernel_size=3, strides=1, kernel_initializer='glorot_uniform', bias_initializer='zeros', padding='same')(X_shortcut)
    X_shortcut = layers.BatchNormalization()(X_shortcut)
    # Identity mapping
    X_identity = layers.UpSampling2D((2,2), data_format="channels_last", name='G_ResBlock_Identity'+str(stage))(X)
    X_identity = layers.Conv2D(num_filters, kernel_size=3, strides=1, kernel_initializer='glorot_uniform', bias_initializer='zeros', padding='same')(X_identity)
    X_identity = layers.BatchNormalization()(X_identity)
    # Concatenate identity and shortcut layers
    X = layers.Add()([X_shortcut, X_identity])
    X = layers.Activation('relu')(X)

    return X

def D_ResBlock(X, down_sample, stage):
    # To input gen_imgs and real_imgs
    # Shortcut layers
    # avg pooling
    X_shortcut = layers.Conv2D(filters=3, kernel_size=(3,3), strides=1, padding='same', kernel_initializer='glorot_uniform', bias_initializer='zeros', name='D_ResBlock_Shortcut'+str(stage))(X)
    X_shortcut = layers.BatchNormalization()(X_shortcut) # Insert spectral normalization here instead
    X_shortcut = layers.ReLU()(X_shortcut)
    X_shortcut = layers.Conv2D(filters=3, kernel_size=(3,3), strides=1, padding='same', kernel_initializer='glorot_uniform', bias_initializer='zeros')(X_shortcut)
    X_shortcut = layers.BatchNormalization()(X_shortcut) # Insert spectral normalization here instead
    if down_sample == True:
        X_shortcut = layers.AveragePooling2D(pool_size=2, padding='same', data_format='channels_last')(X_shortcut) # Downsampling after second conv block "CGANs with Projection Discriminator"
    X_identity = layers.Conv2D(filters=3, kernel_size=(1,1), strides=1, padding='same', kernel_initializer='glorot_uniform', bias_initializer='zeros', name='D_ResBlock_Identity'+str(stage))(X)
    if down_sample == True:
        X_identity = layers.AveragePooling2D(pool_size=2, padding='same', data_format='channels_last')(X_identity) # Downsampling after conv block for identity mapping
    # Concatenate identity and shortcut layers
    X = layers.Add()([X_shortcut, X_identity])
    X = layers.ReLU()(X)
    return X
