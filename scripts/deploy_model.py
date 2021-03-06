import argparse
import os
import sys
from termcolor import colored

import tensorflow as tf
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import graph_io

import keras.backend as K
from keras.models import Model
from keras.models import load_model

from deform_conv.cnn import *
from deform_conv.layers import *
from deform_conv.utils import make_parallel

parser = argparse.ArgumentParser()
parser.add_argument("-w", "--weight", help=".h5 model weight file.")
parser.add_argument("-g", "--gpu_num", help="gpu number used when training this model.")
parser.add_argument("-c", "--class_num", help="number of class of classifier's output")

args = parser.parse_args()
class_num = int(args.class_num) if args.class_num is not None else 6

K.set_learning_phase(0)
inputs, outputs = get_test_cnn(class_num, trainable=False)
# inputs, outputs = get_large_deform_cnn(class_num, trainable=True)
# inputs, outputs = get_large_deform_cnn(class_num, trainable=True)
model = Model(inputs=inputs, outputs=outputs)

if args.gpu_num:
    model = make_parallel(model, int(args.gpu_num))

model.load_weights(args.weight)
model.summary()

if args.gpu_num:
    print(colored('[Deparallelize model]', color='green'))
    single_gpu_model = model.layers[-2]  # TODO: subscript of layers may change?
    model = single_gpu_model
    model.summary()
    model_name = args.weight[:-3] + '_1gpu.h5'
    model.save_weights(model_name)

    print(colored('Restart this script with new .h5 weight:', color='red'))
    print(model_name)
    sys.exit(0)

print(colored('[model.inputs]', color='blue'))
print(model.inputs)
print(colored('[model.outputs]', color='blue'))
print(model.outputs)

new_outputs = []
new_outputs = []
new_outputs_tf = []

if len(model.outputs) > 1:
    for i, n in zip(model.outputs, range(len(model.outputs))):
        new_outputs_tf.append(K.identity(model.outputs[n], name='y_pred_%d' % n))
        new_outputs.append('y_pred_%d' % n)
else:
    new_outputs_tf.append(K.identity(model.outputs[0], name='y_pred'))
    new_outputs.append('y_pred')

# print(colored('[model.inputs]', color='green'))
print(colored('[new model.outputs]', color='green'))
print(new_outputs)

sess = K.get_session()
graph_def = sess.graph.as_graph_def()
names = [n.name for n in graph_def.node]

print('-' * 100)
print(names)
print('Node\'s name contain \"pred\": ')
print([n for n in names if 'pred' in n])
print('-' * 100)

constant_graph = graph_util.convert_variables_to_constants(
    sess, graph_def, new_outputs,
)
# constant_graph = graph_util.remove_training_nodes(constant_graph)
graph_io.write_graph(constant_graph, './backup', 'deploy_model.pb', as_text=False)

print(colored('[PB file saved]', color='green'))
print(os.path.join('./backup', 'deploy_model.pb'))
