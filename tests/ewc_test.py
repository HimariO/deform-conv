from __future__ import division
# %env CUDA_VISIBLE_DEVICES=0
import sys
sys.path.append('../scripts')

import argparse
import os
import random
import numpy as np
import shutil
from termcolor import colored
import tensorflow as tf

import keras as K
from keras.models import Model
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.datasets import fashion_mnist

from deform_conv.callbacks import TensorBoard, SpreadSheet
from deform_conv.cnn import *
from deform_conv.utils import make_parallel
from deform_conv.losses import *

# from dataset_gen import NPZ_gen
from PIL import Image


def get_fashion_mnist():
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

    train_datas = {i:[] for i in range(10)}
    for x, y in zip(x_train, y_train):
        train_datas[y].append(x)

    test_datas = {i:[] for i in range(10)}
    for x, y in zip(x_test, y_test):
        test_datas[y].append(x)

    return train_datas, test_datas


def onehot(id, size=10):
    zero = np.zeros([size])
    zero[id] = 1.
    return zero


def stage_generator(stage, x, batch_size, epoch):
    assert stage < 2
    shif = 5 * stage  # class id offset.
    y = [onehot(0 + shif)] * len(x[0 + shif]) + [onehot(1 + shif)] * len(x[1 + shif]) + [onehot(2 + shif)] * len(x[2 + shif]) + [onehot(3 + shif)] * len(x[3 + shif]) + [onehot(4 + shif)] * len(x[4 + shif])
    x = x[0 + shif] + x[1 + shif] + x[2 + shif] + x[3 + shif] + x[4 + shif]

    for e in range(epoch):
        pairs = list(zip(x, y))
        random.shuffle(pairs)

        for i in range(0, len(pairs), batch_size):
            batch_xy = pairs[i: i + batch_size]
            batch_x = list(map(lambda x: x[0].reshape([28,28,1]), batch_xy))
            batch_y = list(map(lambda x: x[1], batch_xy))
            yield np.array(batch_x), np.array(batch_y)


parser = argparse.ArgumentParser()
parser.add_argument("-g", "--gpu", help="gpu string")
parser.add_argument("-w", "--weight", help=".h5 model weight file.")
# parser.add_argument("-i", "--img_dir", help="feed imgs inside folder.")

args = parser.parse_args()

# ---
# Config
GPU_NUM = 3
GPU = args.gpu

img_size = 28
class_num = 10
batch_size = 32 * GPU_NUM
n_train = (60000) * 1  # Currenly using 200*200 & mtcnn 65*65 dataset

steps_per_epoch = int(np.ceil(n_train / batch_size))
validation_steps = 10000 // batch_size


train_data, test_data = get_fashion_mnist()

task1_train_gen = stage_generator(0, train_data, batch_size, 100)
task2_train_gen = stage_generator(1, train_data, batch_size, 100)

task1_val_gen = stage_generator(0, test_data, 1, 100)
task2_val_gen = stage_generator(1, test_data, 1, 100)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

with tf.Session(config=config) as sess:
    with tf.device(GPU):
        K.backend.set_session(sess)

        # ---
        # Deformable CNN
        inputs, outputs = get_ewc_cnn(class_num)
        model = Model(inputs=inputs, outputs=outputs)

        model.summary()

        # if GPU_NUM > 1:
        #     model = make_parallel(model, GPU_NUM)

        optim = Adam(1e-4)
        # optim = SGD(1e-4, momentum=0.99, nesterov=True)
        loss = categorical_crossentropy
        ewc = EWC(sess, model, loss, DEBUG=True, fisher_multiplier=50)
        loss = ewc.loss
        model.compile(optim, loss, metrics=['accuracy'])

        if args.weight is not None:
            print(colored("[weight] %s" % args.weight, color='green'))
            model.load_weights(args.weight)

            val_loss1, val_acc1 = model.evaluate_generator(
                task1_val_gen, steps=validation_steps
            )

            val_loss2, val_acc2 = model.evaluate_generator(
                task2_val_gen, steps=validation_steps
            )

            print('-' * 100)
            print('task1 loss, acc: ', val_loss1, val_acc1)
            print('task2 loss, acc: ', val_loss2, val_acc2)
            print('-' * 100)

        checkpoint = ModelCheckpoint("cnn_ewc_test.h5", monitor='val_acc', save_best_only=True)
        checkpoint_tl = ModelCheckpoint("cnn_ewc_test_trainbest.h5", monitor='loss', save_best_only=True)

        try:
            total_epoch = 3
            switch_task = 2
            for ii in range(total_epoch):
                print(colored('Round[%d]' % ii, color='green'))

                if ii < switch_task - 1 and os.path.exists('cnn_ewc_task1.h5'):
                    print(colored('SKIP Epoch[%d]' % ii, color='green'))
                    continue

                if ii == switch_task - 1 and os.path.exists('cnn_ewc_task1.h5'):
                    model.load_weights('cnn_ewc_task1.h5')


                train_gen = task1_train_gen if ii < switch_task else task2_train_gen
                val_gen = task1_val_gen if ii < switch_task else task2_val_gen
                train_task = 'task1_train' if ii < switch_task else 'task2_train'

                print('Train on %s' % train_task)
                model.fit_generator(
                    train_gen, steps_per_epoch=steps_per_epoch,
                    epochs=3, verbose=1,
                    validation_data=val_gen, validation_steps=validation_steps,
                    callbacks=[checkpoint, checkpoint_tl]
                )

                if ii == switch_task - 1:
                    model.save_weights('cnn_ewc_task1.h5')
                    ewc.update_fisher(val_gen, batch=batch_size, max_step=6000)
                    loss = ewc.loss
                    model.compile(optim, loss, metrics=['accuracy'])

                val_gen = task1_val_gen
                # val_gen = task1_val_gen if ii >= 5 else task2_val_gen
                val_task = 'task1_val' if ii >= switch_task else 'task2_val'

                val_loss, val_acc = model.evaluate_generator(
                    val_gen, steps=validation_steps
                )

                print('-' * 100)
                print('[%s] val_loss: ' % val_task, val_loss)
                print('[%s] val_acc: ' % val_task, val_acc)
                print('-' * 100)


                if ii == total_epoch - 1:
                    val_loss, val_acc = model.evaluate_generator(
                        task1_val_gen, steps=validation_steps
                    )
                    print('[task1] val_acc: ', val_acc)

                    val_loss, val_acc = model.evaluate_generator(
                        task2_val_gen, steps=validation_steps
                    )
                    print('[task2] val_acc: ', val_acc)

                    ewc.merge_weight()

                    val_loss, val_acc = model.evaluate_generator(
                        task1_val_gen, steps=validation_steps
                    )
                    print('[task1] val_acc: ', val_acc)

                    val_loss, val_acc = model.evaluate_generator(
                        task2_val_gen, steps=validation_steps
                    )
                    print('[task2] val_acc: ', val_acc)

        except KeyboardInterrupt:
            model.save_weights('cnn_ewc_interrupt.h5')
