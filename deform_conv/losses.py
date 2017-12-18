import time
import os
import pickle
import tensorflow as tf
import keras as K
import numpy as np

from keras.layers import Input, Conv2D, SeparableConv2D, Activation, GlobalAvgPool2D, Dense, BatchNormalization, MaxPooling2D, Flatten, Conv2D, Embedding, Lambda
from keras.layers.merge import concatenate, Add
from keras.models import Model

from termcolor import colored

class EWC:
    def __init__(self, sess, model, base_loss, DEBUG=False, fisher_multiplier=2):
        self.sess = sess
        self.model = model
        self.base_loss = base_loss
        self.fisher_multiplier = fisher_multiplier
        # self.F_accum = [0 for i in len(model.trainable_weights)]
        # self.F_accum = np.zeros([len(model.trainable_weights)])
        self.F_accum = None
        self.var_list = model.trainable_weights

        self.DEBUG = DEBUG

    def _print(self, message):
        if self.DEBUG:
            print(colored('[DEBUG] ', color='yellow') + str(message))

    def _save_pickle(self, obj, file_name='fisher_info.pickle'):
        with open(file_name, mode='wb') as f:
            pickle.dump(obj, f)

    def _load_pickle(self, file_name='fisher_info.pickle'):
        with open(file_name, mode='r') as f:
            obj = pickle.load(f)
        return obj

    def update_fisher(self, val_generator, max_step=40, batch=96, Y_is_softmax=True):
        self._print('update_fisher at { %s }' % str(time.time()))

        input_ = model.inputs[0]
        output_ = model.outputs[0]
        self.F_accum = [np.zeros(v.get_shape().as_list()) for v in self.var_list]

        if Y_is_softmax:
            probs = output_
        else:
            probs = tf.nn.softmax(output_)

        class_ind = tf.to_int32(tf.multinomial(tf.log(probs), 1)[0][0])
        ders = tf.gradients(tf.log(probs[0, class_ind]), self.var_list)
        sq_ser = [tf.sqrt(der) for der in ders]
        data_count = 0

        for x, y in val_generator:
            if data_count > max_step:
                break

            f = self.sess.run(sq_ser, feed_dict={input_: x})
            for i, fc in zip(range(len(self.F_accum)), f):
                self.F_accum[i] += fc
            data_count += 1

        # TODO: use nparray will faster
        for i, fc in zip(range(len(self.F_accum)), f):
            self.F_accum[i] /= data_count

        self.star_vars = []
        self.model.save_weights('ewc_star_var.h5')
        self._save_pickle(self.F_accum)

        for v in range(len(self.var_list)):
            self.star_vars.append(self.var_list[v].eval())

    def loss(self, Y_true, Y):
        ewc_loss = self.base_loss(Y_true, Y)

        if self.F_accum is not None:
            # TODO: need a more effice way to clac ewc term, since morden network easy contain >> 1M parameter
            for F in self.F_accum:
                ewc_loss += (fisher_multiplier / 2) * tf.reduce_sum(
                    tf.multiply(
                        self.F_accum[F].astype(np.float32),
                        tf.square(self.var_list[F] - self.star_vars[F])
                    )
                )

        return ewc_loss
