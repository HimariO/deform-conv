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
from dataset_gen_ import *

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

parser = argparse.ArgumentParser()
parser.add_argument("-g", "--gpu", help="gpu string")
parser.add_argument("-w", "--weight", help=".h5 model weight file.")
parser.add_argument("-i", "--img_dir", help="feed imgs inside folder.")
parser.add_argument("-s", "--sample", help="feed imgs inside folder.")

args = parser.parse_args()


# ---
# Config
GPU_NUM = 3
GPU = args.gpu

class_num = 2
batch_size = 32 * GPU_NUM
n_train = 110000 # dataset size
# n_train = 96 * 10 # dataset size
# n_test = batch_size * 10
steps_per_epoch = n_train // batch_size
# steps_per_epoch -= steps_per_epoch % GPU_NUM

validation_steps = (3000 // batch_size)
# validation_steps -= validation_steps % GPU_NUM

dataset = NPZ_gen(
    './person_gender', class_num, batch_size, 1000, resize=224,
    dataset_size=n_train, flip=True, flip_v=True, hierarchy_onehot=False, soft_onehot=False,
    random_scale=0.1, random_crop=0.1, random_resize=None, random_noise=None,
)

train_gen = dataset.get_some()
# val_gen = dataset.get_val(num_batch=validation_steps)
img_data = ImageDataGenerator()
val_gen = img_data.flow_from_directory(
    '/home/share/Ron/a1059s101_1s1_0201',
    target_size=[224, 224],
    batch_size=batch_size,
    shuffle=True
)
validation_steps = 3456//batch_size

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

with tf.Session(config=config) as sess:
    with tf.device(GPU):
        K.set_session(sess)

        # ---
        # Deformable CNN
        inputs, outputs = get_test_cnn(class_num, trainable=True, dropout_sample=args.sample is not None)
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
            try:
                model.fit_generator(
                    train_gen, steps_per_epoch=steps_per_epoch,
                    epochs=1000,
                    validation_data=val_gen, validation_steps=validation_steps,
                    callbacks=[checkpoint, checkpoint_tl, spreadsheet],
                    workers=10, max_queue_size=40,
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
