from __future__ import division
# %env CUDA_VISIBLE_DEVICES=0

import os
import argparse
import sys
import numpy as np
import tensorflow as tf
import keras.backend as K
from termcolor import colored

from keras.models import Model
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import multi_gpu_model

from deform_conv.layers import ConvOffset2D
from deform_conv.callbacks import TensorBoard, SpreadSheet
from deform_conv.cnn import *
from deform_conv.mnist import get_gen
from deform_conv.utils import make_parallel, montecarlo_prediction
# from dataset_gen_ import *

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from keras.datasets import cifar100
from keras.utils import to_categorical

parser = argparse.ArgumentParser()
parser.add_argument("-g", "--gpu", help="gpu string")
parser.add_argument("-w", "--weight", help=".h5 model weight file.")
parser.add_argument("-i", "--img_dir", help="feed imgs inside folder.")
parser.add_argument("-s", "--sample", help="feed imgs inside folder.")

args = parser.parse_args()


# ---
# Config
GPU_NUM = 1
GPU = args.gpu

class_num = 100
batch_size = 32 * GPU_NUM

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

with tf.Session(config=config) as sess:
    with tf.device(GPU):
        K.set_session(sess)

        # ---
        # Deformable CNN
        inputs, outputs = get_scale_cnn(class_num, trainable=True, dropout_sample=args.sample is not None)
        model = Model(inputs=inputs, outputs=outputs)
        model.summary()

        if GPU_NUM > 1:
            model = make_parallel(model, GPU_NUM)
            # model = multi_gpu_model(model, gpus=GPU_NUM)

        if args.weight is not None:
            print(colored("[load_weight] %s" % args.weight, color='green'))
            model.load_weights(args.weight)

        # sys.exit(0)
        # input("Press enter to start training...")
        optim = Adam(1e-4)
        loss = categorical_crossentropy

        model.compile(optim, [loss], metrics=['accuracy'])
        checkpoint = ModelCheckpoint("deform_cnn_pg_best.h5", monitor='val_acc', save_best_only=True)
        checkpoint_tl = ModelCheckpoint("deform_cnn_pg_trainbest.h5", monitor='loss', save_best_only=True)
        spreadsheet = SpreadSheet("1nu6AFqzeYc2rNFAjtUtem-CFYKiRI4HCmXkxWsGglRg", "DeformPersonGenderML3")

        if args.img_dir is None:
            (x_train, y_train), (x_test, y_test) = cifar100.load_data(label_mode='fine')
            # x_train = x_train.astype('float32')
            x_test = x_test.astype('float32')
            # x_train /= 255
            x_test /= 255
            datagen = ImageDataGenerator(
                featurewise_center=True,
                featurewise_std_normalization=True,
                rotation_range=20,
                width_shift_range=0.2,
                height_shift_range=0.2,
                horizontal_flip=True
            )
            datagen.fit(x_train)

            y_train = to_categorical(y_train)
            y_test = to_categorical(y_test)

            try:
                # model.fit(
                #     x=x_train, y=y_train, validation_data=(x_test, y_test),
                #     batch_size=batch_size, epochs=1000,
                #     callbacks=[checkpoint, checkpoint_tl], shuffle=True, initial_epoch=0
                # )
                model.fit_generator(
                    datagen.flow(x_train, y_train, batch_size=32),
                    validation_data=(x_test, y_test), validation_steps=len(x_test) / 32,
                    steps_per_epoch=len(x_train) / 32, epochs=1000,
                    callbacks=[checkpoint, checkpoint_tl],
                )

                model.save_weights('../models/deform_cnn_pg_finish.h5')

            except KeyboardInterrupt:
                model.save_weights('deform_cnn_pg_interrupt.h5')
        else:
            # --
            # Evaluate deformable CNN
            img_data = ImageDataGenerator()

            class_dir = [os.path.join(args.img_dir, d) for d in os.listdir(args.img_dir) if os.path.isdir(os.path.join(args.img_dir, d))]
            img_files = [os.listdir(d) for d in class_dir]

            if args.sample is not None:
                img_gen = img_data.flow_from_directory(
                    args.img_dir,
                    target_size=[200, 200],
                    batch_size=1,
                    shuffle=False
                )
                img_count = len(img_gen.filenames)
                batch_c = 0

                for X, Y in img_gen:
                    if batch_c >= img_count:
                        break
                    prediction_probabilities, prediction_variances = montecarlo_prediction(model, X, 96)

                    print(img_gen.filenames[batch_c], Y)
                    print('prediction_probabilities: ', prediction_probabilities)
                    print('prediction_variances: ', prediction_variances)
                    batch_c += 1

            else:
                img_gen = img_data.flow_from_directory(
                    args.img_dir,
                    target_size=[224, 224],
                    batch_size=batch_size
                )

                val_loss, val_acc = model.evaluate_generator(
                    img_gen, steps=sum(map(lambda x: len(x), img_files)) // batch_size
                )

                print('val_loss:', val_loss)
                print('val_acc:', val_acc)
            # print('Test accuracy of deformable convolution with scaled images', val_acc)
