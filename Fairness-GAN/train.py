# Compiling generator and discriminator here.
from __future__ import print_function, absolute_import, unicode_literals
import tensorflow as tf
import time
import os
import numpy as np
import pandas as pd
import keras
from keras import layers
from keras.optimizers import Adam
from keras.models import Model, Sequential
from keras.utils import plot_model
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from generator import build_generator
from discriminator import build_discriminator

z_dim = 100
nums_classes = 2
img_rows = 64
img_cols = 64
channels = 3
img_shape = (img_rows, img_cols, channels)
attrib_path = 'path/to/list_attr_celeba.csv'
img_path = 'path/to/resized/images'
model_path = 'path/to/models'
saving_path = 'path/to/models'
d_losses = []
g_losses = []
iteration_checkpoints = []
iterations = 20000
batch_size = 64
figure_interval = 5000
plotting_interval = 50
trial = 1
saved_pdf = 'fgan_test_%d.pdf' % trial
pp = PdfPages(saved_pdf)

def build_fgan(generator, discriminator):
    z = layers.Input(shape=(z_dim, ))
    protected_attrib = layers.Input(shape=(1, ))
    img, outcome = generator([z, protected_attrib])
    SJ_XY, SX_X, C_X, C_Y = discriminator([img, outcome])
    model = Model([z, protected_attrib], [SJ_XY, SX_X, C_X, C_Y])

    return model

discriminator = build_discriminator(img_shape, nums_classes)
opt = Adam()
discriminator.compile(loss=['binary_crossentropy', 'binary_crossentropy', 'binary_crossentropy', 'binary_crossentropy'],
        optimizer = opt)
discriminator.trainable = False
generator = build_generator(z_dim, nums_classes)
fgan = build_fgan(generator, discriminator)
fgan.compile(loss=['binary_crossentropy', 'binary_crossentropy', 'binary_crossentropy', 'binary_crossentropy'],
        optimizer = opt)

def sample_images(z_dim, image_grid_rows=2, image_grid_columns=5):
    num_pages = 2
    num_images = num_pages * image_grid_rows * image_grid_columns
    not_attr_male_count = 0
    attr_male_count = 0
    not_attr_female_count = 0
    attr_female_count = 0
    male_image_count = 0        #Test
    female_image_count = 0          #Test
    not_attr_male_imgs = np.array([])
    attr_male_imgs = np.array([])
    not_attr_female_imgs = np.array([])
    attr_female_imgs = np.array([])
    gen_male_outcomes = np.array([])
    gen_female_outcomes = np.array([])
    gen_imgs = np.array([])
    while (not_attr_male_count < 5 or attr_male_count < 5): # Saving 5 images of attractive and non-attractive males
        z = np.random.normal(0, 1, (1, z_dim))
        #print('Value of Z input into generator:', z) # Test
        male = np.ones((1, 1))
        if (not_attr_male_count < 5 or attr_male_count < 5):
            gen_male_img, gen_male_outcome = generator.predict([z, male]) # Generate one male image with its outcome
            gen_male_img = 0.5 * gen_male_img + 0.5 # Scaling image pixels to [0, 1]
            gen_male_outcome = 0.5 * gen_male_outcome + 0.5
            gen_male_img = gen_male_img.astype('float32') # Reducing space usage
            gen_male_outcome = int(np.round(gen_male_outcome, 0))
            male_image_count += 1

            if (gen_male_outcome == 0 and not_attr_male_count < 5): # If generated image outcome is 'Not Attractive' (value=0)
                if not_attr_male_count == 0:
                    not_attr_male_imgs = gen_male_img
                else:
                    not_attr_male_imgs = np.concatenate((not_attr_male_imgs, gen_male_img),  axis=0)
                not_attr_male_count += 1
                print('Not attractive male counts:', not_attr_male_count)
            elif (gen_male_outcome == 1 and attr_male_count < 5): # If generated image outcome is 'Attractive' (value=1)
                if attr_male_count == 0:
                    attr_male_imgs = gen_male_img
                else:
                    attr_male_imgs = np.concatenate((attr_male_imgs, gen_male_img), axis=0)
                attr_male_count += 1

    print('Male images done! [Attractive Male Count: %d] [Not attractive Male Count: %d]' % (attr_male_count, not_attr_male_count))
    while (not_attr_female_count < 5 or attr_female_count < 5):  # Saving 5 images of attractive and non-attractive females
        z = np.random.normal(0, 1, (1, z_dim))
        female = np.repeat(0, 1, axis=0)
        if (not_attr_female_count < 5 or attr_female_count < 5):
            gen_female_img, gen_female_outcome = generator.predict([z, female]) # Generate one female image with its outcome
            gen_female_img = 0.5 * gen_female_img + 0.5 # Scaling image pixels to [0, 1]
            gen_female_outcome = 0.5 * gen_female_outcome + 0.5
            #print('Generated Female outcome before:', gen_female_outcome)
            gen_female_outcome = int(np.round(gen_female_outcome, 0))
            #print('Generated Female outcome after:', gen_female_outcome)
            gen_female_img = gen_female_img.astype('float32')
            female_image_count += 1

            if (gen_female_outcome == 0 and not_attr_female_count < 5): # If generated female image outcome is 'Not Attractive',store image.
                if not_attr_female_count == 0:
                    not_attr_female_imgs = gen_female_img
                else:
                    not_attr_female_imgs = np.concatenate((not_attr_female_imgs, gen_female_img), axis = 0)
                not_attr_female_count += 1

            elif (gen_female_outcome == 1 and attr_female_count < 5): # If generated female image outcome is 'Attractive', store image.
                if attr_female_count == 0:
                    attr_female_imgs = gen_female_img
                else:
                    attr_female_imgs = np.concatenate((attr_female_imgs, gen_female_img), axis = 0)
                attr_female_count += 1
        #print('[Attractive female Count: %d] [Not attractive female Count: %d]' % (attr_female_count, not_attr_female_count))

    print('Female images done! [Attractive Male Count: %d] [Not attractive Male Count: %d]' % (attr_male_count, not_attr_male_count))

    # Combine all the generated images into one array in the order:
    # Male=0, Attractive=0;
    # Male=0, Attractive=1;
    # Male=1, Attractive=0;
    # Male=1, Attractive=1
    gen_imgs = np.concatenate((not_attr_female_imgs, attr_female_imgs, not_attr_male_imgs, attr_male_imgs), axis=0)
    gen_imgs = np.reshape(gen_imgs, (-1, 64, 64, 3))

    try:
    # Plot images
        fig, axs = plt.subplots(image_grid_rows, image_grid_columns, figsize=(10,4), sharey = True, sharex = True)
        cnt = 0
        for page in range(num_pages):
            for i in range(image_grid_rows):
                for j in range(image_grid_columns):
                    axs[i, j].imshow(gen_imgs[cnt, :, :, :])
                    axs[i, j].axis('off')
                    if page == 0:
                        if i == 0:
                            axs[i, j].set_title('male = 0 \n attractive = 0')
                        else:
                            axs[i, j].set_title('male = 0 \n attractive = 1')

                    elif page == 1:
                        if i == 0:
                            axs[i, j].set_title('male = 1 \n attractive = 0')
                        else:
                            axs[i,j].set_title('male = 1 \n attractive = 1')

                    cnt += 1

            pp.savefig()

    except:
        print('Faced an error')
        del not_attr_male_imgs, attr_male_imgs, not_attr_male_count, attr_male_count, not_attr_female_imgs, not_attr_female_count, attr_female_imgs, attr_female_count, female_image_count, male_image_count, z

def train(z_dim, iterations, batch_size, figure_interval, plotting_interval):

    half_batch = int(batch_size/2) # Split X_train into 2 batches to train real images and fake images separately.
    real_half = np.ones((half_batch, 1))
    fake_half = np.zeros((half_batch, 1))
    real = np.ones((batch_size, 1))

    X_train=np.load(img_path)
    X_train=np.reshape(X_train,(-1,64,64,3))
    # Load attributes file
    df = pd.read_csv(attrib_path)

    # Load C_train here.
    C_train = df[['Male']]
    C_train.replace(-1, 0, inplace=True)

    # Load Y_train here.
    Y_train = df[['Attractive']]
    Y_train.replace(-1, 0, inplace=True)

    for iteration in range(iterations):
        start_time = time.time()
        idx = np.random.randint(0, X_train.shape[0], half_batch)
        real_imgs, real_attribs, real_outcomes = X_train[idx], C_train.iloc[idx], Y_train.iloc[idx]
        real_imgs = real_imgs/127.5 - 1.0

        z_real = np.random.normal(0, 1, (half_batch, z_dim))
        _, d_SJ_XY_real, d_SX_X_real, d_C_X_real, d_C_Y_real = discriminator.train_on_batch([real_imgs, real_outcomes], [real_half, real_half, real_attribs, real_attribs])

        z_fake = np.random.normal(0, 1, (half_batch, z_dim))
        gen_imgs, gen_outcomes = generator.predict([z_fake, real_attribs])
        gen_outcomes = np.round(gen_outcomes, 0)
        _, d_SJ_XY_fake, d_SX_X_fake, d_C_X_fake, d_C_Y_fake = discriminator.train_on_batch([gen_imgs,gen_outcomes], [fake_half, fake_half, real_attribs, real_attribs])

        z = np.random.normal(0, 1, (batch_size, z_dim))
        protected_attrib = np.random.randint(0, 2, batch_size).reshape(-1, 1)
        reverse_attrib = flip_value(protected_attrib)    # remember to del at end, to flip values for training C_Y_fake such that they are mutually exclusive
        real = np.ones((batch_size, 1))
        _, g_SJ_XY_fake, g_SX_X_fake, g_C_X_fake, g_C_Y_fake = fgan.train_on_batch([z, protected_attrib], [real, real, protected_attrib, reverse_attrib])

        d_loss = 1/8 * (d_SJ_XY_real + d_SJ_XY_fake + d_SX_X_real + d_SX_X_fake + d_C_X_real + d_C_X_fake + d_C_Y_real + d_C_Y_fake)
        g_loss = 1/4 * (g_SJ_XY_fake + g_SX_X_fake + g_C_X_fake + g_C_Y_fake)

        if (iteration+1) % plotting_interval == 0:
            d_losses.append(d_loss)
            g_losses.append(g_loss)
            iteration_checkpoints.append(iteration+1)

        if (iteration+1) % 1000 == 0:
            end_time = time.time()
            duration = (end_time - start_time)
            print("Iteration: %d [D Loss: %.4f] [G Loss: %.4f] Time taken (seconds) ==> %.4f" %
                        (iteration+1, d_loss, g_loss, duration))

            del start_time, end_time, duration

        if (iteration+1) % figure_interval == 0:
            sample_images(z_dim)
            generator.save(model_path + 'generator_%d_%d.h5' % (iteration+1, trial))
            discriminator.save(model_path + 'discriminator_%d_%d.h5' % (iteration+1, trial))
            fgan.save(model_path + 'fgan_%d_%d.h5' % (iteration+1, trial))

            del reverse_attrib
        if (iteration+1) % 100 == 0:
            generator.save(saving_path + 'generator_latest.h5')
            discriminator.save(saving_path + 'discriminator_latest.h5')
            fgan.save(saving_path + 'fgan_latest.h5')
    df_loss = pd.DataFrame([d_losses, g_losses, iteration_checkpoints], dtype=np.float32)
    df_loss = df_loss.transpose()
    df_loss.columns = ['Discriminator Loss', 'Generator Loss', 'Iteration']
    df_loss.to_excel("Losses_%d.xlsx" % trial,float_format="%.4f")

    del X_train
    plot_losses(iterations=iteration_checkpoints, d_loss=d_losses, g_loss=g_losses)

def plot_losses(iterations, d_loss, g_loss):
    plot = plt.figure()
    plt.plot(iterations, d_loss, label='Discriminator Loss', color='r')
    plt.plot(iterations, g_loss, label='Generator Loss', color='b')
    plt.xlabel('Iterations')
    plt.ylabel('Losses')
    plt.legend(loc = 'lower right')
    plt.draw()
    pp.savefig()

def flip_value(protected_attrib):
    one = np.array([1])
    zero = np.array([0])
    for i in range(len(protected_attrib)):
        if i == 0:
            if protected_attrib[i] == 0:
                reverse_attrib = np.array([1])
            else:
                reverse_attrib = np.array([0])
        else:
            if protected_attrib[i] == 0:
                reverse_attrib = np.concatenate((reverse_attrib, one), axis=0)
            else:
                reverse_attrib = np.concatenate((reverse_attrib, zero), axis=0)

    return reverse_attrib

# Run this function if there are prior models built.
def load_models():
    discriminator.load_weights(model_path + 'discriminator_%d_%d.h5' % (load_iteration,trial-1))
    fgan.load_weights(model_path + 'fgan_%d_%d.h5' % (load_iteration, trial-1))
    generator.load_weights(model_path + 'generator_%d_%d.h5' % (load_iteration, trial-1))
    return discriminator, fgan, generator

train(z_dim=z_dim, iterations=iterations, batch_size=batch_size, figure_interval=figure_interval, plotting_interval=plotting_interval)
pp.close()
