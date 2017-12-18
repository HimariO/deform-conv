from __future__ import division
# %env CUDA_VISIBLE_DEVICES=0

import argparse
import os
import numpy as np
import shutil
from termcolor import colored
import tensorflow as tf

import keras.backend as K
from keras.models import Model
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator

from deform_conv.callbacks import TensorBoard, SpreadSheet
from deform_conv.cnn import *
from deform_conv.utils import make_parallel
from deform_conv.losses import *

from dataset_gen import NPZ_gen
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument("-g", "--gpu", help="gpu string")
parser.add_argument("-w", "--weight", help=".h5 model weight file.")
# parser.add_argument("-i", "--img_dir", help="feed imgs inside folder.")

args = parser.parse_args()

# ---
# Config
GPU_NUM = 3
GPU = args.gpu

img_size = 200
class_num = 6
batch_size = 32 * GPU_NUM
# batch_size = 160 * GPU_NUM #  65*65 img
n_train = (88000 + 100000) * 1  # Currenly using 200*200 & mtcnn 65*65 dataset
# n_test = batch_size * 10
steps_per_epoch = int(np.ceil(n_train / batch_size))
validation_steps = 4000 // batch_size

data_gen_args = dict(featurewise_center=False,
                     featurewise_std_normalization=False,
                     horizontal_flip=True,
                     width_shift_range=0.1,
                     height_shift_range=0.1,
                     zoom_range=0.1)

train_dataset = ImageDataGenerator(**data_gen_args)
train_scaled_gen = train_dataset.flow_from_directory(
    './mtcnn_face_age',
    target_size=[img_size, img_size],
    batch_size=batch_size
)

val_dataset = ImageDataGenerator()
test_scaled_gen = val_dataset.flow_from_directory(
    './mtcnn_face_age_val',
    target_size=[img_size, img_size],
    batch_size=batch_size
)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    with tf.device(GPU):
        K.set_session(sess)

        # ---
        # Deformable CNN
        inputs, outputs = get_cnn(class_num, trainable=True)
        model = Model(inputs=inputs, outputs=outputs)

        model.summary()

        # if GPU_NUM > 1:
        #     model = make_parallel(model, GPU_NUM)

        if args.weight is not None:
            print(colored("[weight] %s" % args.weight, color='green'))
            model.load_weights(args.weight)

        optim = Adam(1e-4)
        # optim = SGD(1e-4, momentum=0.99, nesterov=True)
        loss = categorical_crossentropy
        ewc = EWC(sess, model, loss, DEBUG=True, fisher_multiplier=10)
        checkpoint = ModelCheckpoint("cnn_ewc_test.h5", monitor='val_acc', save_best_only=True)
        checkpoint_tl = ModelCheckpoint("cnn_ewc_test_trainbest.h5", monitor='loss', save_best_only=True)

        try:
            for ii in range(10):
                print('Epoch[%d]' % ii)
                loss = ewc.loss
                model.compile(optim, loss, metrics=['accuracy'])

                train_gen = task1_train_gen if ii < 5 else task2_train_gen
                val_gen = task1_val_gen if ii < 5 else task2_val_gen

                model.fit_generator(
                    train_gen, steps_per_epoch=steps_per_epoch,
                    epochs=1, verbose=1,
                    validation_data=val_gen, validation_steps=validation_steps,
                    callbacks=[checkpoint, checkpoint_tl], workers=5
                )

                ewc.update_fisher(val_gen, batch=batch_size, max_step=100)
                # train_gen = task1_train_gen if ii >= 5 else task2_train_gen
                val_gen = task1_val_gen if ii >= 5 else task2_val_gen

                val_loss, val_acc = model.evaluate_generator(
                    val_gen, steps=validation_steps
                )


        except KeyboardInterrupt:
            model.save_weights('cnn_ewc_interrupt.h5')
