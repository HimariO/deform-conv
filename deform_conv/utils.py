from __future__ import absolute_import, division

from tensorflow.python import debug as tf_debug
import keras.backend as K
from keras.layers import Lambda
from keras.layers.merge import concatenate
from keras.models import load_model, Model
import tensorflow as tf
import numpy as np


def model_save_wrapper(model):
    def load_(file_name, **kwarg):
        if '.np' in file_name:
            model.set_weights(np.load(file_name)['weights'])
        else:
            model.load_weights(file_name, **kwarg)

    def save_(file_name, **kwarg):
        if '.np' in file_name:
            W = model.get_weights()
            np.savez(file_name, weights=W)
        else:
            name_only = file_name.replace('.npy', '')
            name_only = name_only.replace('.npz', '')
            model.save_weights(name_only, **kwarg)

    model.load_weights = load_
    model.save_weights = save_
    return model


def keras_set_tf_debug():
    sess = K.get_session()
    sess = tf_debug.LocalCLIDebugWrapperSession(sess)
    sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)
    K.set_session(sess)

def make_parallel(model, gpu_count):
    def get_slice(data, idx, parts):
        # shape = data.get_shape()
        shape = tf.shape(data)
        #print('[debug][get_slice] ', data)
        #print('[debug][get_slice] ', shape)
        size = tf.concat([shape[:1] // parts, shape[1:]], axis=0)
        stride = tf.concat([shape[:1] // parts, shape[1:] * 0], axis=0)
        #print('[debug][stride] ', stride)
        #print('[debug][size] ', size)
        start = stride * idx
        return tf.slice(data, start, size)

    outputs_all = []
    for i in range(len(model.outputs)):
        outputs_all.append([])

    #Place a copy of the model on each GPU, each getting a slice of the batch

    for i in range(gpu_count):
        with tf.device('/gpu:%d' % i):
            with tf.name_scope('tower_%d' % i) as scope:

                inputs = []
                #Slice each input into a piece for processing on this GPU
                for x in model.inputs:
                    input_shape = tuple(x.get_shape().as_list())[1:]
                    slice_n = Lambda(get_slice, output_shape=input_shape, arguments={'idx':i,'parts':gpu_count})(x)
                    inputs.append(slice_n)
                outputs = model(inputs)

                if not isinstance(outputs, list):
                    outputs = [outputs]
                #Save all the outputs for merging back together later
                for l in range(len(outputs)):
                    outputs_all[l].append(outputs[l])

    # merge outputs on CPU
    with tf.device('/cpu:0'):
        merged = []
        count = 0
        for outputs in outputs_all:
            merged.append(concatenate(outputs, axis=0, name='output_%d' % count))
            count += 1

        return Model(inputs=model.inputs, outputs=merged)


def montecarlo_prediction(model, X_data, T):
    """
    # model - the trained classifier(C classes)
    #                   where the last layer applies softmax
    # X_data - a list of input data(size N)
    # T - the number of monte carlo simulations to run
    """
    # shape: (T, N, C)
    predictions = np.array([model.predict(X_data) for _ in range(T)])

    # shape: (N, C)
    prediction_probabilities = np.mean(predictions, axis=0)

    # shape: (N)
    prediction_variances = np.apply_along_axis(predictive_entropy, axis=1, arr=prediction_probabilities)
    return (prediction_probabilities, prediction_variances)

def predictive_entropy(prob):
    """
    # prob - prediction probability for each class(C). Shape: (N, C)
    # returns - Shape: (N)
    """
    return -1 * np.sum(np.log(prob) * prob, axis=1)
