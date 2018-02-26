from __future__ import division
# %env CUDA_VISIBLE_DEVICES=0
import matplotlib
matplotlib.use('Agg')

import argparse
import os
import numpy as np
import shutil
from termcolor import colored
import tensorflow as tf

import keras.backend as K
from keras.models import Model, load_model
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator

from deform_conv.callbacks import TensorBoard, SpreadSheet
from deform_conv.cnn import *
from deform_conv.utils import make_parallel
from deform_conv.layers import *

from dataset_gen import NPZ_gen
from PIL import Image


from vis.visualization import visualize_saliency, overlay
from vis.utils import utils
from keras import activations
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()
parser.add_argument("-w", "--weight", help=".h5 model weight file.")
parser.add_argument("-i", "--img_dir", help=".h5 model weight file.")
args = parser.parse_args()
class_num = 2
# Build the VGG16 network with ImageNet weights
inputs, outputs = get_large_deform_cnn(class_num, trainable=False)
model = Model(inputs=inputs, outputs=outputs)
# model = get_large_deform_cnn(class_num, trainable=True)
model.load_weights(args.weight)
# model = load_model(args.weight, custom_objects={'InvConv2D': InvConv2D})
print('Model loaded.')


# Utility to search for layer index by name.
# Alternatively we can specify this as -1 since it corresponds to the last layer.

layer_idx = utils.find_layer_idx(model, 'out')
label = [
    '0~20',
    '20~30',
    '30~40',
    '40~50',
    '50~60',
    '60~99',
]

imgs = []

if args.img_dir:
    img_paths = [os.path.join(args.img_dir, i) for i in os.listdir(args.img_dir) if '.jpg' in i or '.png' in i.lower()]
    for imgp in img_paths:
        imgs.append(utils.load_img(imgp, target_size=(200, 200)))
else:
    img1 = utils.load_img('vis_input_img/1.jpg', target_size=(200, 200))
    img2 = utils.load_img('vis_input_img/2.jpg', target_size=(200, 200))
    imgs = [img1, img2]

for i, img in zip(range(len(imgs)), imgs):
    fig = plt.figure()
    fig.add_subplot(3, 3, 1)
    plt.imshow(img)

    # 20 is the imagenet index corresponding to `ouzel`
    for c in range(class_num):
        grads = visualize_saliency(model, layer_idx, filter_indices=c, seed_input=img)
        a = fig.add_subplot(3, 3, 2 + c)
        a.set_title(label[c])
        plt.imshow(grads, cmap='jet')

    # visualize grads as heatmap
    print(i, ': ',grads)
    print('-' * 100)
    # ax[i].imshow(grads, cmap='jet')
    plt.tight_layout()
    fig.savefig('cnn_vis/fig_%d.jpg' % i)
