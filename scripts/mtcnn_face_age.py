from __future__ import division
# %env CUDA_VISIBLE_DEVICES=0

import numpy as np
import tensorflow as tf
import keras.backend as K
from keras.models import Model
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint
from deform_conv.layers import ConvOffset2D
from deform_conv.callbacks import TensorBoard
from deform_conv.cnn import get_large_deform_cnn
from deform_conv.mnist import get_gen
from dataset_gen import NPZ_gen
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
K.set_session(sess)

# ---
# Config
class_num = 6
batch_size = 32
n_train = 90000
n_test = batch_size * 40
steps_per_epoch = int(np.ceil(n_train / batch_size))
validation_steps = int(np.ceil(n_test / batch_size))

dataset = NPZ_gen('./mtcnn_face_age', class_num, batch_size, 1000, dataset_size=n_train)
train_scaled_gen = dataset.get_some()
test_scaled_gen = dataset.get_val(num_batch=40)


# ---
# Deformable CNN

inputs, outputs = get_large_deform_cnn(class_num, trainable=True)
model = Model(inputs=inputs, outputs=outputs)
# model.load_weights('models/cnn.h5', by_name=True)
model.summary()
optim = Adam(5e-4)
# optim = SGD(1e-4, momentum=0.99, nesterov=True)
loss = categorical_crossentropy
model.load_weights('../models/deform_cnn.h5')

model.compile(optim, loss, metrics=['accuracy'])
checkpoint = ModelCheckpoint("deform_cnn_best.h5", monitor='val_acc', save_best_only=True)

model.fit_generator(
    train_scaled_gen, steps_per_epoch=steps_per_epoch,
    epochs=200, verbose=1,
    validation_data=test_scaled_gen, validation_steps=validation_steps,
    callbacks=[checkpoint],
)
# Epoch 20/20
# 1875/1875 [==============================] - 504s - loss: 0.2838 - acc: 0.9122 - val_loss: 0.2359 - val_acc: 0.9231
model.save_weights('../models/deform_cnn.h5')

# --
# Evaluate deformable CNN

model.load_weights('../models/deform_cnn.h5')

val_loss, val_acc = model.evaluate_generator(
    test_scaled_gen, steps=validation_steps
)
print('Test accuracy of deformable convolution with scaled images', val_acc)
# 0.9255

# val_loss, val_acc = model.evaluate_generator(
#     test_gen, steps=validation_steps
# )
# print('Test accuracy of deformable convolution with regular images', val_acc)
# # 0.9727
#
# deform_conv_layers = [l for l in model.layers if isinstance(l, ConvOffset2D)]
#
# Xb, Yb = next(test_gen)
# for l in deform_conv_layers:
#     print(l)
#     _model = Model(inputs=inputs, outputs=l.output)
#     offsets = _model.predict(Xb)
#     offsets = offsets.reshape(offsets.shape[0], offsets.shape[1], offsets.shape[2], -1, 2)
#     print(offsets.min())
#     print(offsets.mean())
#     print(offsets.max())
