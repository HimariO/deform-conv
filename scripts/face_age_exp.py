from __future__ import division
# %env CUDA_VISIBLE_DEVICES=0

import argparse
import os
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
from keras import metrics

from deform_conv.callbacks import TensorBoard, SpreadSheet, NpyCheckPoint
from deform_conv.cnn import *
from deform_conv.NASNet import *
from deform_conv.utils import make_parallel, model_save_wrapper
from deform_conv.losses import *
from deform_conv.nasnet import *
# from keras.layers import GlobalAveragePooling2D, Dropout, Dense, Flatten

from dataset_gen_ import NPZ_gen
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument("-g", "--gpu", help="gpu string")
parser.add_argument("-w", "--weight", help=".h5 model weight file.")
parser.add_argument("-i", "--img_dir", help="feed imgs inside folder.")

args = parser.parse_args()


# ---
# Config
GPU_NUM = 3
GPU = args.gpu

img_size = 200
class_num = 6
batch_size = 128 * GPU_NUM
# batch_size = 160 * GPU_NUM #  65*65 img
n_train = (88000 + 0) * 1  # Currenly using 200*200 & mtcnn 65*65 dataset
# n_train = batch_size * 10
steps_per_epoch = int(np.ceil(n_train / batch_size))
validation_steps = 4000 // batch_size


def main_metric(y_t, y_p):
    return metrics.top_k_categorical_accuracy(y_t, y_p, k=3)


def crop_wrap(generator):
    for x, y in generator:
        new_x = []
        for img in x:
            h, w, c = img.shape
            new_img =  img[int(h * 0.1):int(h * 0.9), int(w * 0.1):int(w * 0.9), :]
            new_x.append(new_img)
        yield np.array(new_x), y


dataset = NPZ_gen(
    './face_age_dataset', class_num, batch_size, 1000,
    dataset_size=n_train, flip=True, hierarchy_onehot=False,
    random_scale=0.2, random_crop=None, random_resize=None, random_noise=0.001,
)

train_scaled_gen = dataset.get_some()
test_scaled_gen = dataset.get_val(num_batch=validation_steps)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

with tf.Session(config=config) as sess:
    with tf.device(GPU):
        # K.backend.set_session(sess)

        # ---
        # Deformable CNN
        inputs, outputs = get_cnn(class_num, trainable=True)
        # inputs, outputs = NASNet(class_num, trainable=True, img_size=200)
        model = Model(inputs=inputs, outputs=outputs)
        # print(colored("[model_inputs]", color='green'), inputs)
        # print(colored("[model_outputs]", color='green'), outputs)


        # model = Model(inputs=model.inputs, outputs=[x])


        model.summary()
        if GPU_NUM > 1:
            model = make_parallel(model, GPU_NUM)
        # model.summary()

        model = model_save_wrapper(model)

        if args.weight is not None:
            print(colored("[weight] %s" % args.weight, color='green'))
            model.load_weights(args.weight)

        optim = Adam(1e-4)
        loss = categorical_crossentropy
        # loss = PGCE

        model.compile(optim, [loss], metrics=['accuracy'])
        # checkpoint = ModelCheckpoint("deform_cnn_exp_best.npz", monitor='val_acc', save_best_only=False)
        # checkpoint_tl = ModelCheckpoint("deform_cnn_exp_trainbest.npz", monitor='loss', save_best_only=False)
        checkpoint = NpyCheckPoint(model, "deform_cnn_exp_best.npz", monitor='val_acc')
        checkpoint_tl = NpyCheckPoint(model, "deform_cnn_exp_trainbest.npz", monitor='loss')
        spreadsheet = SpreadSheet("1nu6AFqzeYc2rNFAjtUtem-CFYKiRI4HCmXkxWsGglRg", "DeformFaceAgeML3")

        if args.img_dir is None:
            try:
                model.fit_generator(
                    train_scaled_gen, steps_per_epoch=steps_per_epoch,
                    epochs=1000, verbose=1,
                    validation_data=test_scaled_gen, validation_steps=validation_steps,
                    callbacks=[checkpoint, checkpoint_tl, spreadsheet], workers=12, max_queue_size=36,
                )

                val_loss, val_acc = model.evaluate_generator(
                    test_scaled_gen, steps=validation_steps
                )

                print('Test accuracy of deformable convolution with scaled images', val_acc)
            except KeyboardInterrupt:
                model.save_weights('deform_cnn_exp_interrupt.npz')
                # np.savez('deform_cnn_exp_interrupt.npz', weights=model.get_weights())
        else:
            if True:

                img_data = ImageDataGenerator()
                img_gen = img_data.flow_from_directory(
                    args.img_dir,
                    target_size=[img_size, img_size],
                    batch_size=batch_size
                )

                class_dir = [os.path.join(args.img_dir, d) for d in os.listdir(args.img_dir) if os.path.isdir(os.path.join(args.img_dir, d))]
                img_files = [os.listdir(d) for d in class_dir]

                val_loss, val_acc = model.evaluate_generator(
                    crop_wrap(img_gen), steps=sum(map(lambda x: len(x), img_files)) // batch_size
                )

                print('val_loss:', val_loss)
                print('val_acc:', val_acc)
            else:
                img_dir = [os.path.join(args.img_dir, d) for d in os.listdir(args.img_dir) if not os.path.isdir(os.path.join(args.img_dir, d))]
                inputs = []

                if os.path.exists('./test_result'):
                    shutil.rmtree('./test_result')

                os.mkdir('./test_result')
                for i in range(class_num):
                    os.mkdir('./test_result/%d' % i)

                c = 0
                counter = {
                    0: 0,
                    1: 0,
                    2: 0,
                    3: 0,
                    4: 0,
                    5: 0,
                }

                # for img_p in img_dir:
                #     pil_img = Image.open(img_p)
                #     pil_img = pil_img.resize([img_size, img_size])
                #     np_img = np.asarray(pil_img, dtype=np.uint8)
                #     # np_img /= 255
                #     inputs.append(np_img)

                for x, y in test_scaled_gen:
                    for yy in y:
                        counter[yy.argmax()] += 1
                    if c > validation_steps:
                        break
                    inputs += [np_i for np_i in x]
                    # print(inputs)
                    # input()
                    c += 1
                print(counter)

                c = 0
                pred_counter = {
                    0: 0,
                    1: 0,
                    2: 0,
                    3: 0,
                    4: 0,
                    5: 0,
                }

                # for x, y in test_scaled_gen:
                #     if c > validation_steps:
                #         break
                #     pred = model.predict(x, batch_size=batch_size)
                #     for yy in pred:
                #         pred_counter[yy.argmax()] += 1
                #     c += 1
                for i in range(0, len(inputs), batch_size):
                    if i + batch_size > len(inputs):
                        break
                    batch = np.array(inputs[i: i + batch_size], dtype=np.float32)
                    pred = model.predict(batch, batch_size=batch_size)

                    for o, n in zip(pred, range(len(pred))):
                        # print(o.shape)
                        class_id = o.argmax()
                        pred_counter[class_id] += 1
                        Image.fromarray(inputs[i + n].astype(np.uint8)).save('./test_result/%d/%d.jpg' % (class_id, i + n))
                print(pred_counter)
