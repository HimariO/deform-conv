import time
import os
import pickle
import tensorflow as tf
import keras as K
import numpy as np

from keras.layers import Input, Conv2D, SeparableConv2D, Activation, GlobalAvgPool2D, Dense, BatchNormalization, MaxPooling2D, Flatten, Conv2D, Embedding, Lambda
from keras.layers.merge import concatenate, Add
from keras.models import Model
from keras.losses import categorical_crossentropy

from termcolor import colored

class EWC:
    def __init__(self, sess, model, base_loss, DEBUG=False,
                 fisher_multiplier=10, Y_is_softmax=True, model_in=None, model_out=None):
        self.DEBUG = DEBUG
        self.sess = sess
        self.model = model
        assert len(model.outputs) == 1

        self.base_loss = base_loss
        self.fisher_multiplier = fisher_multiplier
        # self.F_accum = [0 for i in len(model.trainable_weights)]
        # self.F_accum = np.zeros([len(model.trainable_weights)])
        self.F_accum = None
        self.var_list = model.trainable_weights
        self._print(self.var_list)

        self._build_fisher_graph(Y_is_softmax, model_in=model_in, model_out=model_out)

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

    def _build_fisher_graph(self, Y_is_softmax, model_in=None, model_out=None):
        self.input_ = input_ = self.model.inputs[0] if model_in is None else model_in
        self.output_ = output_ = self.model.outputs[0] if model_out is None else model_out
        # self.target = target = tf.placeholder(tf.float32, shape=[None, output_.get_shape().as_list()[1]])  # onehot target

        if Y_is_softmax:
            probs = output_
        else:
            probs = tf.nn.softmax(output_)

        """
        need to determine which output to use when we clac fisher info.
        use entire Y will cause fisher info loss be to strong to let network to learn anything new.
        but random sample single output class want best soultion either, performance of original task
        drop pretty sinficanly after train on new task.
        use ground true class is probably the best option!
        """
        # class_ind = tf.to_int32(tf.multinomial(tf.log(probs), 1)[0][0])
        class_ind = tf.argmax(probs, axis=1)

        # self._debug_ders = batch_prob = tf.log(probs)
        # self._debug_ders = tf.gather(probs, class_ind)
        # self._debug_ders = class_ind
        self._debug_ders = tf.reduce_max(probs, axis=1)
        # batch_prob = tf.log(tf.reduce_sum(probs))  # use onehot target as loss weight
        batch_prob = tf.log(tf.reduce_max(probs, axis=1))  # use onehot target as loss weight
        # batch_prob = tf.reduce_sum(batch_prob)
        ders = tf.gradients(batch_prob, self.var_list)
        sq_der = [tf.square(der) for der in ders]
        self.sq_der = sq_der
        # raise NotImplemented("!")

    def update_fisher(self, val_generator, max_step=40, batch=96):
        self._print('update_fisher at { %s }' % str(time.time()))
        self.F_accum = [np.zeros(v.get_shape().as_list()) for v in self.var_list]
        data_count = 0

        for x, y in val_generator:
            # TODO: this version of implmentation has fixed batch 1. need some upgrade in the fature.
            # for n in range(len(x)):
            if data_count > max_step:
                break

            f, _f = self.sess.run([self.sq_der, self._debug_ders], feed_dict={
                    K.backend.learning_phase(): 0,
                    self.input_: x,
                })
            # self._print(f)
            # input()

            for i, fc in zip(range(len(self.F_accum)), f):
                self.F_accum[i] += fc
            data_count += x.shape[0]

        # TODO: use nparray will faster
        for i in range(len(self.F_accum)):
            self.F_accum[i] /= data_count

        self.star_vars = []
        self.model.save_weights('ewc_star_var.h5')
        self._save_pickle(self.F_accum)

        for v in range(len(self.var_list)):
            self.star_vars.append(self.var_list[v].eval())

        # self._print(self.F_accum)

    def loss(self, Y_true, Y):
        ewc_loss = self.base_loss(Y_true, Y)

        if self.F_accum is not None:
            # TODO: need a more effice way to clac ewc term, since morden network easy contain >> 1M parameter

            self._print("Using ewc loss terms!")
            for v in range(len(self.var_list)):
                ewc_loss += (self.fisher_multiplier / 2) * tf.reduce_sum(
                    tf.multiply(
                        self.F_accum[v].astype(np.float32),
                        tf.square(self.var_list[v] - self.star_vars[v])
                    )
                )

        return ewc_loss

    def merge_weight(self, last_weight=None):
        assert self.F_accum is not None, 'Need fisher infomation of trainable variable to do this!'
        assert last_weight != None or len(self.star_vars) > 0

        for i in range(len(self.var_list)):
            f_info = self.F_accum[i]
            flat_f = f_info.reshape([-1])
            flat_f.sort()
            gate = flat_f[int(len(flat_f) / 4 * 2.5)]
            self._print(gate)
            f_mask = np.greater(f_info, gate).astype(np.float32)
            # self._print(f_mask)

            star = self.star_vars[i]
            star_masked = star * f_mask
            # self._print(star_masked)

            var = self.var_list[i].eval()
            var_masked = var * np.abs(f_mask - 1)
            # self._print(var_masked)

            new_var = var_masked + star_masked
            self.sess.run(tf.assign(self.var_list[i], new_var))
            self._print('[%s] Insert %d/%d Variable' % (self.var_list[i].name, f_mask.sum(), np.prod(f_mask.shape)))


def PGCE(y_true, y_pred):
    CE = categorical_crossentropy(y_true, y_pred)
    PG = tf.multiply(
        tf.abs(y_true - tf.reduce_max(y_true)),
        (y_pred)
    )
    PG = tf.reduce_sum(PG, axis=1)
    PG = tf.reduce_mean(PG, axis=0)
    return CE + PG
