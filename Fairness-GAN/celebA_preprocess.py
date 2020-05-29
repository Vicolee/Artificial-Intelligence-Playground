# To preprocess images and its respective attributes
# Create a new directory for resized images.
# Convert resized images into numpy arrays for training.
# Create a numpy array of shape (total number of images, total number of attributes)
import os
import numpy as np
import pandas as pd
from PIL import Image
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
import matplotlib.pyplot as plt

data_dir = 'path/to/img_align_celeba/'
save_dir = 'path/to/resized_img/'

image_height = 64
image_width = 64
num_channels = 3
X = []
C = []

if not os.path.isdir(save_dir):
    os.mkdir(save_dir)

img_list = os.listdir(data_dir)
img_list.sort()

for i in range(1000):
    img = Image.open(data_dir + img_list[i])
    img = img.resize((image_height, image_width), Image.LANCZOS) # Use the lanczos filter. Smoothens image.
    img.save(save_dir + img_list[i],  'JPEG')

    if i % 1000 == 0:
        print('Resizing %d images...' % i)

# Load resized images into numpy arrays
for i in range(1000):
    img = load_img(save_dir+img_list[i])
    #img_array = img_to_array(img, data_format='channels_last')
    if i == 0:
        X = np.array(img)
    else:
        img_array = np.array(img)
        X = np.append(X, img_array, axis = 0)

# Reshaping the images into (num_images, num_rows, num_cols, num_channels)
X = np.reshape(X,(-1, image_height, image_width, num_channels))

#Processing protected attributes (Gender in this case) (labelled C as in Fairness GAN paper)
attr_path = "path/to/list_attr_celeba.csv"
attr_df = pd.read_csv(attr_path)
C = attr_df[['image_id','Male']] # Choosing Male as protected attribute
pd.to_numeric(C['Male'])
C['Male'].replace({-1:0}, inplace=True)

# Processing outcomes (Attractiveness in this case) (labelled Y as in Fairness GAN paper)
Y = attr_df[['image_id', 'Attractive']]
pd.to_numeric(Y['Attractive'])
Y['Attractive'].replace({-1:0}, inplace=True)

