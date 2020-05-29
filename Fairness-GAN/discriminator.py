from resblock import D_ResBlockSN, D_ResBlock
from spectral_normalization import DenseSN,ConvSN2D
import tensorflow as tf
from keras import layers
from keras.models import Model, Sequential
import numpy as np
from keras.utils import plot_model

def build_discriminator(img_shape, nums_classes):
    img = layers.Input(name='X_img', shape=img_shape)
    model = layers.Conv2D(64, kernel_size=(3,3), strides = 1, activation='relu', kernel_initializer='glorot_uniform', bias_initializer='zeros', padding='same')(img)            #Check
    model = D_ResBlockSN(X=img, down_sample=True, stage=1)
    model = D_ResBlockSN(X=model, down_sample=True, stage=2)
    model = D_ResBlockSN(X=model, down_sample=True, stage=3)
    model = D_ResBlockSN(X=model, down_sample=True, stage=4) # 4x4x1024

    # Probability that Source X is correctly identified given image X(real|fake)
    SX_X = layers.Flatten()(model)
    SX_X = DenseSN(name='SX_X_Linear', units=128, activation='relu', kernel_initializer='glorot_uniform', bias_initializer='zeros')(SX_X)
    SX_X = DenseSN(name='SX_X', units=1, activation='sigmoid')(SX_X)

    # Probability of C (protected attribute) given image X
    C_X = layers.Flatten()(model)
    C_X = DenseSN(name='C_X_Linear', units=128, kernel_initializer='glorot_uniform', bias_initializer='zeros', activation='relu')(C_X)
    C_X = DenseSN(name='C_X', units=1, activation='sigmoid')(C_X)

    # Probability of Source J given outcome Y(real|fake)
    y = layers.Input(name='Y_outcome', shape=(1,), dtype='float32')
    y_embed = layers.Embedding(2, 4*4*1024, input_length=1)(y) # outputs:1x128
    y_flattened = layers.Flatten()(y_embed)
    y_reshaped = layers.Reshape((4,4,1024))(y_flattened)
    concatenated = layers.Concatenate(axis=-1)([model, y_reshaped])
    SJ_XY = ConvSN2D(512, kernel_size=(3,3), strides=1, padding='same', activation='relu')(concatenated)
    SJ_XY = layers.Flatten()(SJ_XY)
    SJ_XY = DenseSN(units=128, activation='relu')(SJ_XY)
    SJ_XY = DenseSN(name='SJ_XY', units=1, activation='sigmoid')(SJ_XY)

    # Probability of C (protected attribute) given Y
    C_Y = DenseSN(name='C_Y_Linear_1', units=128, kernel_initializer='glorot_uniform', activation='relu')(y)
    C_Y = DenseSN(name='C_Y_Linear_2', units=128, kernel_initializer='glorot_uniform', activation='relu')(C_Y)
    C_Y = DenseSN(name='C_Y', units=1, kernel_initializer='glorot_uniform', activation='sigmoid')(C_Y)

    discriminator = Model(inputs=[img, y], outputs=[SJ_XY, SX_X, C_X, C_Y])

    return discriminator
