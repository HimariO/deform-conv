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
    def __init__(self, sess, model, base_loss, DEBUG=False, fisher_multiplier=10, Y_is_softmax=True):
        self.sess = sess
        self.model = model
        assert len(model.outputs) == 1

        self.base_loss = base_loss
        self.fisher_multiplier = fisher_multiplier
        # self.F_accum = [0 for i in len(model.trainable_weights)]
        # self.F_accum = np.zeros([len(model.trainable_weights)])
        self.F_accum = None
        self.var_list = model.trainable_weights

        self.DEBUG = DEBUG
        self._build_fisher_graph(Y_is_softmax)

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

    def _build_fisher_graph(self, Y_is_softmax):
        self.input_ = input_ = self.model.inputs[0]
        self.output_ = output_ = self.model.outputs[0]
        self.target = target = tf.placeholder(tf.float32, shape=[None, output_.get_shape().as_list()[1]])  # onehot target

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

        batch_prob = tf.log(tf.gather(probs, class_ind))  # use onehot target as loss weight
        batch_prob = tf.reduce_sum(batch_prob)
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

            f = self.sess.run(self.sq_der, feed_dict={
                    K.backend.learning_phase(): 0,
                    self.input_: x,
                    self.target: y,
                })
            # self._print(f)

            for i, fc in zip(range(len(self.F_accum)), f):
                self.F_accum[i] += fc
            data_count += 1

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
            # self._print(self.F_accum)

            self._print("Using ewc loss terms!")
            for v in range(len(self.var_list)):
                ewc_loss += (self.fisher_multiplier / 2) * tf.reduce_sum(
                    tf.multiply(
                        self.F_accum[v].astype(np.float32),
                        tf.square(self.var_list[v] - self.star_vars[v])
                    )
                )

        return ewc_loss


def PGCE(y_true, y_pred):
    # print('-' * 100)
    # print(y_true.get_shape().as_list())
    # print(y_pred.get_shape().as_list())
    # print('-' * 100)
    CE = categorical_crossentropy(y_true, y_pred)
    PG = tf.multiply(tf.abs(y_true - 1), y_pred)
    PG = tf.reduce_sum(PG, axis=1)
    PG = tf.reduce_mean(PG, axis=0)
    return CE + PG
