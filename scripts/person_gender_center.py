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
from deform_conv.losses import PGCE

from dataset_gen_ import NPZ_gen, NPZ_class_id
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
class_num = 2
batch_size = 32 * GPU_NUM
# batch_size = 160 * GPU_NUM #  65*65 img
n_train = (73152) * 1  # Currenly using 200*200 & mtcnn 65*65 dataset
# n_train = 300
steps_per_epoch = int(np.ceil(n_train / batch_size))
validation_steps = 3000 // batch_size


def val_wrapper(generator):
    for x, y in generator:
        batch_class_id = np.zeros([x.shape[0]])
        batch_center = np.zeros([x.shape[0], 256])
        yield [x, batch_class_id], [y, batch_center]

dataset = NPZ_class_id(
    './person_gender_ikea', class_num, batch_size, 1000,
    dataset_size=n_train, flip=True, hierarchy_onehot=False, soft_onehot=False,
    random_scale=0.1, random_crop=0.1, random_resize=None, random_noise=None,
)
train_gen = dataset.get_some()
val_gen = dataset.get_val(num_batch=validation_steps)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    K.set_session(sess)

    # ---
    # Deformable CNN
    model = deform_center_cnn(class_num, trainable=True, GPU=GPU_NUM)
    # model = Model(inputs=inputs, outputs=outputs)
    model.summary()

    if args.weight is not None:
        print(colored("[weight] %s" % args.weight, color='green'))
        model.load_weights(args.weight)

    optim = Adam(1e-4)
    # optim = SGD(1e-4, momentum=0.99, nesterov=True)
    loss = PGCE
    # model._weights('../models/deform_cnn.h5')

    model.compile(optim, loss=[loss, lambda y_true, y_pred: K.sum(y_pred)], loss_weights=[1., .001], metrics={'output': 'accuracy'})
    checkpoint = ModelCheckpoint("pg_center_best.h5", monitor='val_output_acc', save_best_only=True)
    checkpoint_tl = ModelCheckpoint("pg_center_trainbest.h5", monitor='output_loss', save_best_only=True)
    spreadsheet = SpreadSheet("1nu6AFqzeYc2rNFAjtUtem-CFYKiRI4HCmXkxWsGglRg", "DeformFaceAgeML3")

    if args.img_dir is None:
        try:
            model.fit_generator(
                train_gen, steps_per_epoch=steps_per_epoch,
                epochs=1000, verbose=1,
                validation_data=val_gen, validation_steps=validation_steps,
                callbacks=[checkpoint, checkpoint_tl],
            )

            val_loss, val_acc = model.evaluate_generator(
                val_gen, steps=validation_steps
            )

            print('Test accuracy of deformable convolution with scaled images', val_acc)
        except KeyboardInterrupt:
            model.save_weights('pg_center_interrupt.h5')
    else:
        if True:
            img_data = ImageDataGenerator(rescale=1./255)
            img_gen = img_data.flow_from_directory(
                args.img_dir,
                target_size=[img_size, img_size],
                batch_size=batch_size
            )
            img_gen = val_wrapper(img_gen)

            class_dir = [os.path.join(args.img_dir, d) for d in os.listdir(args.img_dir) if os.path.isdir(os.path.join(args.img_dir, d))]
            img_files = [os.listdir(d) for d in class_dir]

            val_loss, v2_l, val_acc, v2_a = model.evaluate_generator(
                img_gen, steps=sum(map(lambda x: len(x), img_files)) // batch_size
            )

            print('val_loss:', v2_l)
            print('val_acc:', v2_a)
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
