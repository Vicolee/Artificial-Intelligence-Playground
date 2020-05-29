from resblock import *
from keras.optimizers import Adam
from keras.initializers import RandomNormal
from keras.utils import plot_model
from keras import Model, Sequential
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras import layers

def build_generator(z_dim,nums_classes):
    z = layers.Input(shape=(z_dim,), name='Noise_Vector_128_dim')
    protected_attrib = layers.Input(shape=(1, ), name='Protected_Attribute', dtype='int32')
    Attrib_embedding = layers.Embedding(nums_classes, z_dim, input_length = 1)(protected_attrib)
    Attrib_embedding = layers.Flatten(name='Flatten')(Attrib_embedding)
    joined_representation = layers.Multiply(name='Joined_Representation')([z, Attrib_embedding])

    # Image model
    image_model = layers.Dense(name='Image_Path_Dense_1', units=1024*4*4, activation='relu', input_dim=z_dim, kernel_initializer='glorot_uniform', bias_initializer='zeros')(joined_representation)
    image_model = layers.Reshape((4, 4, 1024))(image_model)
    image_model=G_ResBlock(X=image_model,stage=1)
    image_model=G_ResBlock(X=image_model,stage=2)
    image_model=G_ResBlock(X=image_model,stage=3)
    image_model=G_ResBlock(X=image_model,stage=4) # (64,64,128)
    gen_imgs = layers.Conv2D(name='Image_Path_gen_img', filters=3, kernel_size=(3,3), strides=1, activation='tanh', padding='same', kernel_initializer='glorot_uniform', bias_initializer='zeros')(image_model)

    # Outcome model
    outcome_model = layers.Dense(name='Outcome_Path_Dense_1', units=1024*4*4, input_dim=z_dim, kernel_initializer = 'glorot_uniform', bias_initializer='zeros')(joined_representation)
    outcome_model = layers.BatchNormalization()(outcome_model)
    outcome_model = layers.Activation('tanh')(outcome_model)
    outcome_model = layers.Dense(name='Outcome_Path_Dense_2', units=128*4*4, kernel_initializer = 'glorot_uniform', bias_initializer='zeros')(outcome_model)
    outcome_model = layers.BatchNormalization()(outcome_model)
    outcome_model = layers.Activation('tanh')(outcome_model)
    gen_outcomes = layers.Dense(name='Outcome_Path_gen_outcome', units=1, activation = 'sigmoid', kernel_initializer='glorot_uniform', bias_initializer='zeros')(outcome_model)

    return Model([z, protected_attrib], [gen_imgs, gen_outcomes])
