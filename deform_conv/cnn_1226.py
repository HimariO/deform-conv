from __future__ import absolute_import, division

import keras as K
from keras.layers import Input, Conv2D, SeparableConv2D, Activation, GlobalAvgPool2D, Dense, BatchNormalization, MaxPooling2D, Flatten, Conv2D, Embedding, Lambda
from keras.layers.merge import concatenate, Add
from keras.models import Model
from deform_conv.layers import *
from deform_conv.utils import make_parallel

from keras.initializers import Orthogonal, lecun_normal


def get_cnn(class_n):
	inputs = l = Input((None, None, 1), name='input')

	# conv11
	l = Conv2D(32, (3, 3), padding='same', name='conv11')(l)
	l = Activation('relu', name='conv11_relu')(l)
	l = BatchNormalization(name='conv11_bn')(l)

	# conv12
	l = Conv2D(64, (3, 3), padding='same', strides=(2, 2), name='conv12')(l)
	l = Activation('relu', name='conv12_relu')(l)
	l = BatchNormalization(name='conv12_bn')(l)

	# conv21
	l = Conv2D(128, (3, 3), padding='same', name='conv21')(l)
	l = Activation('relu', name='conv21_relu')(l)
	l = BatchNormalization(name='conv21_bn')(l)

	# conv22
	l = Conv2D(128, (3, 3), padding='same', strides=(2, 2), name='conv22')(l)
	l = Activation('relu', name='conv22_relu')(l)
	l = BatchNormalization(name='conv22_bn')(l)

	# out
	l = GlobalAvgPool2D(name='avg_pool')(l)
	l = Dense(class_n, name='fc1')(l)
	outputs = l = Activation('softmax', name='out')(l)

	return inputs, outputs


def get_deform_cnn(trainable):
	inputs = l = Input((28, 28, 1), name='input')

	# conv11
	l = Conv2D(32, (3, 3), padding='same', name='conv11', trainable=trainable)(l)
	l = Activation('relu', name='conv11_relu')(l)
	l = BatchNormalization(name='conv11_bn')(l)

	# conv12
	l_offset = ConvOffset2D(32, name='conv12_offset')(l)
	l = Conv2D(64, (3, 3), padding='same', strides=(2, 2), name='conv12', trainable=trainable)(l_offset)
	l = Activation('relu', name='conv12_relu')(l)
	l = BatchNormalization(name='conv12_bn')(l)

	# conv21
	l_offset = ConvOffset2D(64, name='conv21_offset')(l)
	l = Conv2D(128, (3, 3), padding='same', name='conv21', trainable=trainable)(l_offset)
	l = Activation('relu', name='conv21_relu')(l)
	l = BatchNormalization(name='conv21_bn')(l)

	# conv22
	l_offset = ConvOffset2D(128, name='conv22_offset')(l)
	l = Conv2D(128, (3, 3), padding='same', strides=(2, 2), name='conv22', trainable=trainable)(l_offset)
	l = Activation('relu', name='conv22_relu')(l)
	l = BatchNormalization(name='conv22_bn')(l)

	# out
	l = GlobalAvgPool2D(name='avg_pool')(l)
	l = Dense(10, name='fc1', trainable=trainable)(l)
	outputs = l = Activation('softmax', name='out')(l)

	return inputs, outputs

def get_large_deform_cnn(class_num, trainable=False):
	# init = Orthogonal(gain=1.0, seed=None)
	init = lecun_normal()

	inputs = l = Input((200, 200, 3), name='input')

	norm_input = ImageNorm()(inputs)
	# norm_input = inputs

	# conv11
	l = Conv2D(32, (3, 3), padding='same', name='conv11', trainable=trainable, kernel_initializer=init, kernel_regularizer=OrthLocalReg2D)(norm_input)
	l = Activation('selu', name='conv11_relu')(l)
	l = BatchNormalization(name='conv11_bn', center=False, scale=False)(l)

	l2 = InvConv2D(32, (3, 3), padding='same', name='inv_conv11', trainable=trainable, kernel_initializer=init, kernel_regularizer=OrthLocalReg2D)(norm_input)
	l2 = Activation('selu', name='inv_conv11_relu')(l2)
	l2 = BatchNormalization(name='inv_conv11_bn', center=False, scale=False)(l2)

	l3 = Conv2D(32, (3, 1), padding='same', name='conv11_2', trainable=trainable, kernel_initializer=init, kernel_regularizer=OrthLocalReg2D)(norm_input)
	l3 = Activation('selu', name='conv11_2_relu')(l3)
	l3 = BatchNormalization(name='conv11_2_bn', center=False, scale=False)(l3)

	l5 = Conv2D(32, (1, 3), padding='same', name='conv11_3', trainable=trainable, kernel_initializer=init, kernel_regularizer=OrthLocalReg2D)(norm_input)
	l5 = Activation('selu', name='conv11_3_relu')(l5)
	l5 = BatchNormalization(name='conv11_3_bn', center=False, scale=False)(l5)

	l = concatenate([l, l2, l3, l5])

	# conv12
	# l_offset = ConvOffset2D(32, name='conv12_offset')(l)

	l = Conv2D(128, (3, 3), padding='same', strides=(2, 2), name='pool11_12', trainable=trainable, kernel_initializer=init, kernel_regularizer=OrthLocalReg2D)(l)
	l = Activation('selu', name='pool11_12_relu')(l)
	# l = BatchNormalization(name='conv12_bn', center=False, scale=False)(l)


	l3 = Conv2D(32, (3, 1), padding='same', name='conv12_2', trainable=trainable, kernel_initializer=init, kernel_regularizer=OrthLocalReg2D)(l)
	l3 = Activation('selu', name='conv12_2_relu')(l3)
	l3 = BatchNormalization(name='conv12_2_bn', center=False, scale=False)(l3)

	l5 = Conv2D(32, (1, 3), padding='same', name='conv12_3', trainable=trainable, kernel_initializer=init, kernel_regularizer=OrthLocalReg2D)(l)
	l5 = Activation('selu', name='conv12_3_relu')(l5)
	l5 = BatchNormalization(name='conv12_3_bn', center=False, scale=False)(l5)

	l = Conv2D(128, (3, 3), padding='same', name='conv12', trainable=trainable, kernel_initializer=init, kernel_regularizer=OrthLocalReg2D)(l)
	l = Activation('selu', name='conv12_relu')(l)
	l = BatchNormalization(name='conv12_bn', center=False, scale=False)(l)

	l = concatenate([l, l3, l5])

	l = Conv2D(128, (3, 3), padding='same', name='conv13', trainable=trainable, kernel_initializer=init, kernel_regularizer=OrthLocalReg2D)(l)
	l = Activation('selu', name='conv13_relu')(l)
	l = BatchNormalization(name='conv13_bn', center=False, scale=False)(l)

	l = Conv2D(128, (3, 3), padding='same', strides=(2, 2), name='conv14', trainable=trainable, kernel_initializer=init, kernel_regularizer=OrthLocalReg2D)(l)
	l = Activation('selu', name='conv14_relu')(l)
	l = BatchNormalization(name='conv14_bn', center=False, scale=False)(l)

	# l = Conv2D(192, (3, 3), padding='same', name='conv21', trainable=trainable, kernel_initializer=init, kernel_regularizer=OrthLocalReg2D)(l)
	# l = Activation('selu', name='conv21_relu')(l)
	# l = BatchNormalization(name='conv21_bn', center=False, scale=False)(l)
	#
	# l = Conv2D(192, (3, 3), padding='same', name='conv22', trainable=trainable, kernel_initializer=init, kernel_regularizer=OrthLocalReg2D)(l)
	# l = Activation('selu', name='conv22_relu')(l)
	# l = BatchNormalization(name='conv22_bn', center=False, scale=False)(l)

	# conv22
	# l_offset = ConvOffset2D(192, name='conv32_offset')(l)
	# l = Conv2D(192, (3, 3), padding='same', strides=(2, 2), name='conv23', trainable=trainable, kernel_initializer=init, kernel_regularizer=OrthLocalReg2D)(l)
	# l = Activation('selu', name='conv23_relu')(l)
	# l = BatchNormalization(name='conv23_bn', center=False, scale=False)(l)

	l = Conv2D(256, (3, 3), padding='same', strides=(2, 2), name='conv31', trainable=trainable, kernel_initializer=init, kernel_regularizer=OrthLocalReg2D)(l)
	l = Activation('selu', name='conv31_relu')(l)
	l31 = BatchNormalization(name='conv31_bn', center=False, scale=False)(l)

	l = Conv2D(256, (1, 1), padding='same', name='conv32', trainable=trainable, kernel_initializer=init, kernel_regularizer=OrthLocalReg2D)(l31)
	l = Activation('selu', name='conv32_relu')(l)
	l = BatchNormalization(name='conv32_bn', center=False, scale=False)(l)

	l = Conv2D(256, (3, 3), padding='same', name='conv33', trainable=trainable, kernel_initializer=init, kernel_regularizer=OrthLocalReg2D)(l)
	l = Activation('selu', name='conv33_relu')(l)
	l = BatchNormalization(name='conv33_bn', center=False, scale=False)(l)
	l = Add(name='residual_31_32')([l31, l])


	l_offset = ConvOffset2D(256, name='conv33_offset')(l)
	l = Conv2D(512, (3, 3), padding='same', name='conv41', trainable=trainable, kernel_initializer=init, kernel_regularizer=OrthLocalReg2D)(l_offset)
	l = Activation('selu', name='conv41_relu')(l)
	l = BatchNormalization(name='conv41_bn', center=False, scale=False)(l)

	l = Conv2D(512, (3, 3), padding='same', strides=(2, 2), name='conv42', trainable=trainable, kernel_initializer=init, kernel_regularizer=OrthLocalReg2D)(l)
	l = Activation('selu', name='conv42_relu')(l)
	l = BatchNormalization(name='conv42_bn', center=False, scale=False)(l)

	# l_offset = ConvOffset2D(512, name='conv35_offset')(l)
	l = Conv2D(512, (3, 3), padding='same', name='conv43', trainable=trainable, kernel_initializer=init, kernel_regularizer=OrthLocalReg2D)(l)
	l = Activation('selu', name='conv43_relu')(l)
	l = BatchNormalization(name='conv43_bn', center=False, scale=False)(l)

	l = Conv2D(1024, (3, 3), padding='same', strides=(2, 2), name='conv51', trainable=trainable, kernel_initializer=init, kernel_regularizer=OrthLocalReg2D)(l)
	l = Activation('selu', name='conv43_relu')(l)
	l = BatchNormalization(name='conv43_bn', center=False, scale=False)(l)

	l = Conv2D(1024, (3, 3), padding='same', name='conv52', trainable=trainable, kernel_initializer=init, kernel_regularizer=OrthLocalReg2D)(l)
	l = Activation('selu', name='conv43_relu')(l)
	l = BatchNormalization(name='conv43_bn', center=False, scale=False)(l)

	# out
	# l = GlobalAvgPool2D(name='avg_pool')(l)
	l = MaxPooling2D(name='max_pool_final')(l)
	l = Flatten(name='flatten_maxpool')(l)
	l = Dense(768, name='fc1', trainable=trainable, kernel_initializer=init)(l)
	l = Activation('selu', name='fc1_relu')(l)

	l = Dense(256, name='fc2', trainable=trainable, kernel_initializer=init)(l)
	l = Activation('selu', name='fc2_relu')(l)

	l = Dense(class_num, name='fc3', trainable=trainable)(l)
	outputs = l = Activation('softmax', name='out')(l)

	return inputs, outputs


def get_large_deform_inv_cnn(class_num, trainable=False):
	inputs = l = Input((200, 200, 3), name='input')

	# conv11
	l = Conv2D(32, (3, 3), padding='same', name='conv11', trainable=trainable, kernel_regularizer=OrthLocalReg2D)(l)
	l = Activation('relu', name='conv11_relu')(l)
	l = BatchNormalization(name='conv11_bn')(l)

	l2 = InvConv2D(32, (3, 3), padding='same', name='inv_conv11', trainable=trainable, kernel_regularizer=OrthLocalReg2D)(inputs)
	l2 = Activation('relu', name='inv_conv11_relu')(l2)
	l2 = BatchNormalization(name='inv_conv11_bn')(l2)

	l3 = Conv2D(32, (3, 5), padding='same', name='conv11_2', trainable=trainable, kernel_regularizer=OrthLocalReg2D)(inputs)
	l3 = Activation('relu', name='conv11_2_relu')(l3)
	l3 = BatchNormalization(name='conv11_2_bn')(l3)

	l5 = Conv2D(32, (5, 3), padding='same', name='conv11_3', trainable=trainable, kernel_regularizer=OrthLocalReg2D)(inputs)
	l5 = Activation('relu', name='conv11_3_relu')(l5)
	l5 = BatchNormalization(name='conv11_3_bn')(l5)

	l = concatenate([l, l2, l3, l5])

	# conv12
	# l_offset = ConvOffset2D(32, name='conv12_offset')(l)

	l = Conv2D(128, (3, 3), padding='same', strides=(2, 2), name='pool11_12', trainable=trainable, kernel_regularizer=OrthLocalReg2D)(l)
	l = Activation('relu', name='pool11_12_relu')(l)
	# l = BatchNormalization(name='conv12_bn')(l)


	l3 = Conv2D(32, (3, 5), padding='same', name='conv12_2', trainable=trainable, kernel_regularizer=OrthLocalReg2D)(l)
	l3 = Activation('relu', name='conv12_2_relu')(l3)
	l3 = BatchNormalization(name='conv12_2_bn')(l3)

	l5 = Conv2D(32, (5, 3), padding='same', name='conv12_3', trainable=trainable, kernel_regularizer=OrthLocalReg2D)(l)
	l5 = Activation('relu', name='conv12_3_relu')(l5)
	l5 = BatchNormalization(name='conv12_3_bn')(l5)

	l = Conv2D(128, (3, 3), padding='same', name='conv12', trainable=trainable, kernel_regularizer=OrthLocalReg2D)(l)
	l = Activation('relu', name='conv12_relu')(l)
	l = BatchNormalization(name='conv12_bn')(l)
	l_block2 = l

	l = concatenate([l, l3, l5])

	l = Conv2D(128, (3, 3), padding='same', name='conv13', trainable=trainable, kernel_regularizer=OrthLocalReg2D)(l)
	l = Activation('relu', name='conv13_relu')(l)
	l = BatchNormalization(name='conv13_bn')(l)

	l = Conv2D(128, (3, 3), padding='same', strides=(2, 2), name='conv14', trainable=trainable, kernel_regularizer=OrthLocalReg2D)(l)
	l = Activation('relu', name='conv14_relu')(l)
	l = BatchNormalization(name='conv14_bn')(l)

	# l = Conv2D(192, (3, 3), padding='same', name='conv21', trainable=trainable, kernel_regularizer=OrthLocalReg2D)(l)
	# l = Activation('relu', name='conv21_relu')(l)
	# l = BatchNormalization(name='conv21_bn')(l)

	# l = Conv2D(192, (3, 3), padding='same', name='conv22', trainable=trainable, kernel_regularizer=OrthLocalReg2D)(l)
	# l = Activation('relu', name='conv22_relu')(l)
	# l = BatchNormalization(name='conv22_bn')(l)

	# conv22
	# l_offset = ConvOffset2D(192, name='conv32_offset')(l)
	l = Conv2D(192, (3, 3), padding='same', name='conv23', trainable=trainable, kernel_regularizer=OrthLocalReg2D)(l)
	l = Activation('relu', name='conv23_relu')(l)
	l = BatchNormalization(name='conv23_bn')(l)

	l24 = Conv2D(192, (7, 7), padding='same', strides=(2, 2), name='conv24', trainable=trainable, kernel_regularizer=OrthLocalReg2D)(l_block2)
	l24 = Activation('relu', name='conv24_relu')(l24)
	l24 = BatchNormalization(name='conv24_bn')(l24)

	l = concatenate([l ,l24])

	l = Conv2D(256, (3, 3), padding='same', name='conv31', trainable=trainable, kernel_regularizer=OrthLocalReg2D)(l)
	l = Activation('relu', name='conv31_relu')(l)
	l31 = BatchNormalization(name='conv31_bn')(l)

	l = Conv2D(256, (3, 3), padding='same', name='conv32', trainable=trainable, kernel_regularizer=OrthLocalReg2D)(l31)
	l = Activation('relu', name='conv32_relu')(l)
	l = BatchNormalization(name='conv32_bn')(l)

	l = Conv2D(256, (3, 3), padding='same', name='conv33', strides=(2, 2), trainable=trainable, kernel_regularizer=OrthLocalReg2D)(l)
	l = Activation('relu', name='conv33_relu')(l)
	l = BatchNormalization(name='conv33_bn')(l)
	# l = Add(name='residual_31_32')([l31, l])


	l_offset = ConvOffset2D(256, name='conv33_offset')(l)
	l = Conv2D(512, (3, 3), padding='same', name='conv41', trainable=trainable, kernel_regularizer=OrthLocalReg2D)(l_offset)
	l = Activation('relu', name='conv41_relu')(l)
	l = BatchNormalization(name='conv41_bn')(l)

	l = Conv2D(512, (3, 3), padding='same', strides=(2, 2), name='conv42', trainable=trainable, kernel_regularizer=OrthLocalReg2D)(l)
	l = Activation('relu', name='conv42_relu')(l)
	l = l_res = BatchNormalization(name='conv42_bn')(l)

	# l_offset = ConvOffset2D(512, name='conv35_offset')(l)
	l = Conv2D(512, (3, 3), padding='same', name='conv43', trainable=trainable, kernel_regularizer=OrthLocalReg2D)(l)
	l = Activation('relu', name='conv43_relu')(l)
	l = BatchNormalization(name='conv43_bn')(l)

	l = Conv2D(512, (3, 3), padding='same', name='conv44', trainable=trainable, kernel_regularizer=OrthLocalReg2D)(l)
	l = Activation('relu', name='conv44_relu')(l)
	l = BatchNormalization(name='conv44_bn')(l)

	l = Add(name='residual_42_44')([l_res, l])

	l = MaxPooling2D(name='max_pool_5')(l)

	l = Conv2D(1024, (3, 3), padding='same', name='conv51', trainable=trainable, kernel_regularizer=OrthLocalReg2D)(l)
	l = Activation('selu', name='conv51_relu')(l)
	l = BatchNormalization(name='conv51_bn', center=False, scale=False)(l)

	l = Conv2D(1024, (3, 3), padding='same', name='conv52', trainable=trainable, kernel_regularizer=OrthLocalReg2D)(l)
	l = Activation('selu', name='conv52_relu')(l)
	l = BatchNormalization(name='conv52_bn', center=False, scale=False)(l)

	# out
	# l = MaxPooling2D(name='max_pool_final')(l)
	# l = Flatten(name='flatten_maxpool')(l)

	l = SpatialPyramidPooling([1, 2, 4], input_shape=[None, None, 512])(l)
	l = Dense(768, name='fc1', trainable=trainable, kernel_regularizer=OrthLocalReg1D)(l)
	l = Activation('relu', name='fc1_relu')(l)

	l = Dense(256, name='fc2', trainable=trainable, kernel_regularizer=OrthLocalReg1D)(l)
	l = Activation('relu', name='fc2_relu')(l)

	l = Dense(class_num, name='fc3', trainable=trainable)(l)
	outputs = l = Activation('softmax', name='out')(l)

	return inputs, outputs


def get_large_res_deform_cnn(class_num, trainable=False):
	inputs = l = Input((200, 200, 3), name='input')

	"""
	Block 1
	"""
	l = Conv2D(32, (3, 3), padding='same', name='conv11', trainable=trainable, kernel_regularizer=OrthLocalReg2D)(l)
	l = Activation('relu', name='conv11_relu')(l)
	l = BatchNormalization(name='conv11_bn')(l)

	l2 = InvConv2D(32, (3, 3), padding='same', name='inv_conv11', trainable=trainable, kernel_regularizer=OrthLocalReg2D)(inputs)
	l2 = Activation('relu', name='inv_conv11_relu')(l2)
	l2 = BatchNormalization(name='inv_conv11_bn')(l2)

	l3 = Conv2D(32, (3, 1), padding='same', name='conv11_2', trainable=trainable, kernel_regularizer=OrthLocalReg2D)(inputs)
	l3 = Activation('relu', name='conv11_2_relu')(l3)
	l3 = BatchNormalization(name='conv11_2_bn')(l3)

	l5 = Conv2D(32, (1, 3), padding='same', name='conv11_3', trainable=trainable, kernel_regularizer=OrthLocalReg2D)(inputs)
	l5 = Activation('relu', name='conv11_3_relu')(l5)
	l5 = BatchNormalization(name='conv11_3_bn')(l5)

	l = concatenate([l, l2, l3, l5])

	# conv12
	# l_offset = ConvOffset2D(32, name='conv12_offset')(l)

	l = SeparableConv2D(128, (3, 3), padding='same', strides=(2, 2), name='pool11_12', trainable=trainable, depthwise_regularizer=OrthLocalRegSep2D)(l)
	l = Activation('relu', name='pool11_12_relu')(l)
	# l = BatchNormalization(name='conv11_12_bn')(l)

	"""
	Block 2
	"""

	l3 = SeparableConv2D(128, (3, 1), padding='same', name='conv12_2', trainable=trainable)(l)
	l3 = Activation('relu', name='conv12_2_relu')(l3)
	l3 = BatchNormalization(name='conv12_2_bn')(l3)

	l5 = SeparableConv2D(128, (1, 3), padding='same', name='conv12_3', trainable=trainable)(l)
	l5 = Activation('relu', name='conv12_3_relu')(l5)
	l5 = BatchNormalization(name='conv12_3_bn')(l5)

	l = SeparableConv2D(128, (3, 3), padding='same', name='conv12', trainable=trainable, depthwise_regularizer=OrthLocalRegSep2D)(l)
	l = Activation('relu', name='conv12_relu')(l)
	l = BatchNormalization(name='conv12_bn')(l)

	l = concatenate([l, l3, l5])

	"""
	Normal Convs
	"""

	l = Conv2D(192, (3, 3), padding='same', name='conv13', trainable=trainable, kernel_regularizer=OrthLocalReg2D)(l)
	l = Activation('relu', name='conv13_relu')(l)
	l = BatchNormalization(name='conv13_bn')(l)

	l = SeparableConv2D(192, (3, 3), padding='same', strides=(2, 2), name='conv14', trainable=trainable, depthwise_regularizer=OrthLocalRegSep2D)(l)
	l = Activation('relu', name='conv14_relu')(l)
	# l = BatchNormalization(name='conv14_bn')(l)

	l = SeparableConv2D(192, (3, 3), padding='same', name='conv21', trainable=trainable, depthwise_regularizer=OrthLocalRegSep2D)(l)
	l = Activation('relu', name='conv21_relu')(l)
	l_res_h = BatchNormalization(name='conv21_bn')(l)

	l = SeparableConv2D(192, (3, 3), padding='same', name='conv22', trainable=trainable, depthwise_regularizer=OrthLocalRegSep2D)(l_res_h)
	l = Activation('relu', name='conv22_relu')(l)
	l = BatchNormalization(name='conv22_bn')(l)
	l = Add(name='residual_21_23')([l_res_h, l])

	# conv22
	# l_offset = ConvOffset2D(192, name='conv32_offset')(l)
	l = SeparableConv2D(192, (3, 3), padding='same', strides=(2, 2), name='conv23', trainable=trainable, depthwise_regularizer=OrthLocalRegSep2D)(l)
	l = Activation('relu', name='conv23_relu')(l)
	l = BatchNormalization(name='conv23_bn')(l)

	l = SeparableConv2D(256, (3, 3), padding='same', name='conv31', trainable=trainable, depthwise_regularizer=OrthLocalRegSep2D)(l)
	l = Activation('relu', name='conv31_relu')(l)
	l_res_h = BatchNormalization(name='conv31_bn')(l)

	l = SeparableConv2D(256, (3, 3), padding='same', name='conv32', trainable=trainable, depthwise_regularizer=OrthLocalRegSep2D)(l_res_h)
	l = Activation('relu', name='conv32_relu')(l)
	l = BatchNormalization(name='conv32_bn')(l)

	l = SeparableConv2D(256, (3, 3), padding='same', name='conv33', trainable=trainable, depthwise_regularizer=OrthLocalRegSep2D)(l)
	l = Activation('relu', name='conv33_relu')(l)
	l = BatchNormalization(name='conv33_bn')(l)
	l = Add(name='residual_31_33')([l_res_h, l])


	l_offset = ConvOffset2D(256, name='conv33_offset')(l)
	l = Conv2D(256, (3, 3), padding='same', name='conv41', trainable=trainable)(l_offset)
	l = Activation('relu', name='conv41_relu')(l)
	l = BatchNormalization(name='conv41_bn')(l)

	l = SeparableConv2D(512, (3, 3), padding='same', strides=(2, 2), depth_multiplier=2, name='conv42', trainable=trainable, depthwise_regularizer=OrthLocalRegSep2D)(l)
	l = Activation('relu', name='conv42_relu')(l)
	l_res_h = BatchNormalization(name='conv42_bn')(l)

	# l_offset = ConvOffset2D(512, name='conv35_offset')(l)
	l = SeparableConv2D(512, (3, 3), padding='same', name='conv43', trainable=trainable, depthwise_regularizer=OrthLocalRegSep2D)(l_res_h)
	l = Activation('relu', name='conv43_relu')(l)
	l = BatchNormalization(name='conv43_bn')(l)

	l = SeparableConv2D(512, (3, 3), padding='same', name='conv44', trainable=trainable, depthwise_regularizer=OrthLocalRegSep2D)(l)
	l = Activation('relu', name='conv44_relu')(l)
	l = BatchNormalization(name='conv44_bn')(l)
	l = Add(name='residual_42_44')([l_res_h, l])

	# out
	# l = GlobalAvgPool2D(name='avg_pool')(l)
	# l = MaxPooling2D(name='max_pool_final')(l)

	# l = SeparableConv2D(512, (3, 3), padding='same', name='conv51', trainable=trainable, kernel_regularizer=OrthLocalRegSep2D)(l)
	# l = Activation('relu', name='conv51_relu')(l)
	# l = BatchNormalization(name='conv51_bn')(l)

	l = SpatialPyramidPooling([1, 2, 4], input_shape=[None, None, 512])(l)

	# l = Flatten(name='flatten_maxpool')(l)
	l = Dense(768, name='fc1', trainable=trainable, kernel_regularizer=OrthLocalReg1D)(l)
	l = Activation('relu', name='fc1_relu')(l)

	l = Dense(256, name='fc2', trainable=trainable)(l)
	l = Activation('relu', name='fc2_relu')(l)

	l = Dense(class_num, name='fc3', trainable=trainable)(l)
	outputs = l = Activation('softmax', name='out')(l)

	return inputs, outputs


def get_large_deform_cnn2(class_num, trainable=False, GPU=1):
	# init = Orthogonal(gain=1.0, seed=None)
	init = 'random_normal'

	inputs = l = Input((200, 200, 3), name='input')
	input_target = Input((1,), name='input_target')

	#norm_input = ImageNorm()(inputs)
	norm_input = inputs

	# conv11
	l = Conv2D(32, (3, 3), padding='same', name='conv11', trainable=trainable, kernel_initializer=init, kernel_regularizer=OrthLocalReg2D)(norm_input)
	l = Activation('relu', name='conv11_relu')(l)
	l = BatchNormalization(name='conv11_bn')(l)

	l2 = InvConv2D(32, (3, 3), padding='same', name='inv_conv11', trainable=trainable, kernel_initializer=init, kernel_regularizer=OrthLocalReg2D)(norm_input)
	l2 = Activation('relu', name='inv_conv11_relu')(l2)
	l2 = BatchNormalization(name='inv_conv11_bn')(l2)

	l3 = Conv2D(32, (3, 1), padding='same', name='conv11_2', trainable=trainable, kernel_initializer=init, kernel_regularizer=OrthLocalReg2D)(norm_input)
	l3 = Activation('relu', name='conv11_2_relu')(l3)
	l3 = BatchNormalization(name='conv11_2_bn')(l3)

	l5 = Conv2D(32, (1, 3), padding='same', name='conv11_3', trainable=trainable, kernel_initializer=init, kernel_regularizer=OrthLocalReg2D)(norm_input)
	l5 = Activation('relu', name='conv11_3_relu')(l5)
	l5 = BatchNormalization(name='conv11_3_bn')(l5)

	l4 = InvConv2D(32, (3, 1), padding='same', name='conv11_2i', trainable=trainable, kernel_initializer=init, kernel_regularizer=OrthLocalReg2D)(norm_input)
	l4 = Activation('relu', name='conv11_2i_relu')(l4)
	l4 = BatchNormalization(name='conv11_2i_bn')(l4)

	l6 = InvConv2D(32, (1, 3), padding='same', name='conv11_3i', trainable=trainable, kernel_initializer=init, kernel_regularizer=OrthLocalReg2D)(norm_input)
	l6 = Activation('relu', name='conv11_3i_relu')(l6)
	l6 = BatchNormalization(name='conv11_3i_bn')(l6)

	l = concatenate([l, l2, l3, l5, l4, l6])

	# conv12
	# l_offset = ConvOffset2D(32, name='conv12_offset')(l)

	l5 = Conv2D(128, (5, 5), padding='same', strides=(2, 2), name='pool5_11_12', trainable=trainable, kernel_initializer=init, kernel_regularizer=OrthLocalReg2D)(l)
	l5 = Activation('relu', name='pool5_11_12_relu')(l5)
	l5 = BatchNormalization(name='pool5_11_12_bn')(l5)

	l3 = Conv2D(128, (3, 3), padding='same', strides=(2, 2), name='pool3_11_12', trainable=trainable, kernel_initializer=init, kernel_regularizer=OrthLocalReg2D)(l)
	l3 = Activation('relu', name='pool3_11_12_relu')(l3)
	l3 = BatchNormalization(name='pool3_11_12_bn')(l3)

	l = concatenate([l3, l5])

	l3 = Conv2D(32, (5, 3), padding='same', name='conv12_2', trainable=trainable, kernel_initializer=init, kernel_regularizer=OrthLocalReg2D)(l)
	l3 = Activation('relu', name='conv12_2_relu')(l3)
	l3 = BatchNormalization(name='conv12_2_bn')(l3)

	l5 = Conv2D(32, (3, 5), padding='same', name='conv12_3', trainable=trainable, kernel_initializer=init, kernel_regularizer=OrthLocalReg2D)(l)
	l5 = Activation('relu', name='conv12_3_relu')(l5)
	l5 = BatchNormalization(name='conv12_3_bn')(l5)

	l = Conv2D(128, (3, 3), padding='same', name='conv12', trainable=trainable, kernel_initializer=init, kernel_regularizer=OrthLocalReg2D)(l)
	l = Activation('relu', name='conv12_relu')(l)
	l = BatchNormalization(name='conv12_bn')(l)

	l = concatenate([l, l3, l5])

	l = Conv2D(128, (3, 3), padding='same', name='conv13', trainable=trainable, kernel_initializer=init, kernel_regularizer=OrthLocalReg2D)(l)
	l = Activation('relu', name='conv13_relu')(l)
	l = jump = BatchNormalization(name='conv13_bn')(l)

	l = Conv2D(192, (3, 3), padding='same', strides=(2, 2), name='conv14', trainable=trainable, kernel_initializer=init, kernel_regularizer=OrthLocalReg2D)(l)
	l = Activation('relu', name='conv14_relu')(l)
	# l = BatchNormalization(name='conv14_bn')(l)

	l = Conv2D(192, (3, 3), padding='same', name='conv21', trainable=trainable, kernel_initializer=init, kernel_regularizer=OrthLocalReg2D)(l)
	l = Activation('relu', name='conv21_relu')(l)
	l = BatchNormalization(name='conv21_bn')(l)

	l = Conv2D(192, (3, 3), padding='same', name='conv22', trainable=trainable, kernel_initializer=init, kernel_regularizer=OrthLocalReg2D)(l)
	l = Activation('relu', name='conv22_relu')(l)
	l = BatchNormalization(name='conv22_bn')(l)

	# conv22
	# l_offset = ConvOffset2D(192, name='conv32_offset')(l)
	l = Conv2D(256, (3, 3), padding='same', strides=(2, 2), name='conv23', trainable=trainable, kernel_initializer=init, kernel_regularizer=OrthLocalReg2D)(l)
	l = Activation('relu', name='conv23_relu')(l)
	l = BatchNormalization(name='conv23_bn')(l)

	l = Conv2D(256, (3, 3), padding='same', name='conv31', trainable=trainable, kernel_initializer=init, kernel_regularizer=OrthLocalReg2D)(l)
	l = Activation('relu', name='conv31_relu')(l)
	l31 = BatchNormalization(name='conv31_bn')(l)

	# l = Conv2D(256, (1, 1), padding='same', name='conv32', trainable=trainable, kernel_initializer=init, kernel_regularizer=OrthLocalReg2D)(l31)
	# l = Activation('relu', name='conv32_relu')(l)
	# l = BatchNormalization(name='conv32_bn')(l)

	l = Conv2D(256, (3, 3), padding='same', name='conv33', trainable=trainable, kernel_initializer=init, kernel_regularizer=OrthLocalReg2D)(l)
	l = Activation('relu', name='conv33_relu')(l)
	l = BatchNormalization(name='conv33_bn')(l)
	l = Add(name='residual_31_32')([l31, l])


	lj = Conv2D(256, (3, 3), padding='same', strides=(2, 2), name='jump_pool', trainable=trainable, kernel_initializer=init, kernel_regularizer=OrthLocalReg2D)(jump)
	lj = Activation('relu', name='jump_pool_relu')(lj)
	lj = BatchNormalization(name='jump_pool_bn')(lj)

	lj = Conv2D(256, (3, 3), padding='same', strides=(2, 2), name='jump_pool2', trainable=trainable, kernel_initializer=init, kernel_regularizer=OrthLocalReg2D)(lj)
	lj = Activation('relu', name='jump_pool2_relu')(lj)
	lj = BatchNormalization(name='jump_pool2_bn')(lj)

	l = concatenate([l, lj])

	l_offset = ConvOffset2D(512, name='conv33_offset')(l)
	l = Conv2D(512, (3, 3), padding='same', strides=(2, 2), name='conv41', trainable=trainable, kernel_initializer=init, kernel_regularizer=OrthLocalReg2D)(l_offset)
	l = Activation('relu', name='conv41_relu')(l)
	l = BatchNormalization(name='conv41_bn')(l)

	l = Conv2D(512, (3, 3), padding='same', name='conv42', trainable=trainable, kernel_initializer=init, kernel_regularizer=OrthLocalReg2D)(l)
	l = Activation('relu', name='conv42_relu')(l)
	l = BatchNormalization(name='conv42_bn')(l)

	# l_offset = ConvOffset2D(512, name='conv35_offset')(l)
	l = Conv2D(512, (3, 3), padding='same', name='conv43', trainable=trainable, kernel_initializer=init, kernel_regularizer=OrthLocalReg2D)(l)
	l = Activation('relu', name='conv43_relu')(l)
	l = BatchNormalization(name='conv43_bn')(l)

	# out
	# l = GlobalAvgPool2D(name='avg_pool')(l)
	l = MaxPooling2D(name='max_pool_final')(l)
	l = Flatten(name='flatten_maxpool')(l)
	l = Dense(768, name='fc1', trainable=trainable, kernel_initializer=init, kernel_regularizer=OrthLocalReg1D)(l)
	l = Activation('relu', name='fc1_relu')(l)

	l = feature = Dense(256, name='fc2', trainable=trainable, kernel_initializer=init)(l)
	l = Activation('relu', name='fc2_relu')(l)

	l = Dense(class_num, name='fc3', trainable=trainable)(l)
	outputs = l = Activation('softmax', name='out')(l)


	if GPU == 1 :
		centers = Embedding(class_num, 256)(input_target)
		l2_loss = Lambda(lambda x: K.sum(K.square(x[0] - x[1][:, 0]), 1, keepdims=True), name='l2_loss')([feature, centers])
		Model(inputs=[inputs, input_target], outputs=[outputs, l2_loss])
	elif GPU == 0:
		return inputs, outputs
	else:
		BODY = Model(inputs=[inputs, input_target], outputs=[outputs, feature])
		BODY = make_parallel(BODY, GPU)
		softmax_output = Lambda(lambda x: x, name='output')(BODY.outputs[0])

		centers = Embedding(class_num, 256)(input_target)
		l2_loss = Lambda(lambda x: K.sum(K.square(x[0] - x[1][:, 0]), 1, keepdims=True), name='l2_loss')([BODY.outputs[1], centers])
		model_withcneter = Model(inputs=BODY.inputs, outputs=[softmax_output, l2_loss])
		return model_withcneter

	# return [inputs, input_target], [outputs, l2_loss]


def get_large_res_deform_cnn2(class_num, trainable=False):
	inputs = l = Input((None, None, 3), name='input')

	"""
	Block 1
	"""
	l = Conv2D(32, (3, 3), padding='same', name='conv11', trainable=trainable, kernel_regularizer=OrthLocalReg2D)(l)
	l = Activation('relu', name='conv11_relu')(l)
	l = BatchNormalization(name='conv11_bn')(l)

	l2 = InvConv2D(32, (3, 3), padding='same', name='inv_conv11', trainable=trainable, kernel_regularizer=OrthLocalReg2D)(inputs)
	l2 = Activation('relu', name='inv_conv11_relu')(l2)
	l2 = BatchNormalization(name='inv_conv11_bn')(l2)

	l3 = Conv2D(32, (5, 5), padding='same', name='conv11_2', trainable=trainable, kernel_regularizer=OrthLocalReg2D)(inputs)
	l3 = Activation('relu', name='conv11_2_relu')(l3)
	l3 = BatchNormalization(name='conv11_2_bn')(l3)

	l5 = InvConv2D(32, (5, 5), padding='same', name='conv11_3', trainable=trainable, kernel_regularizer=OrthLocalReg2D)(inputs)
	l5 = Activation('relu', name='conv11_3_relu')(l5)
	l5 = BatchNormalization(name='conv11_3_bn')(l5)

	l = concatenate([l, l2, l3, l5])

	# conv12
	# l_offset = ConvOffset2D(32, name='conv12_offset')(l)

	l = SeparableConv2D(128, (3, 3), padding='same', strides=(2, 2), name='pool11_12', trainable=trainable, depthwise_regularizer=OrthLocalRegSep2D)(l)
	l = Activation('relu', name='pool11_12_relu')(l)
	# l = BatchNormalization(name='conv11_12_bn')(l)

	"""
	Block 2
	"""

	l3 = SeparableConv2D(128, (5, 3), padding='same', name='conv12_2', trainable=trainable)(l)
	l3 = Activation('relu', name='conv12_2_relu')(l3)
	l3 = BatchNormalization(name='conv12_2_bn')(l3)

	l5 = SeparableConv2D(128, (3, 5), padding='same', name='conv12_3', trainable=trainable)(l)
	l5 = Activation('relu', name='conv12_3_relu')(l5)
	l5 = BatchNormalization(name='conv12_3_bn')(l5)

	l = SeparableConv2D(128, (3, 3), padding='same', name='conv12', trainable=trainable, depthwise_regularizer=OrthLocalRegSep2D)(l)
	l = Activation('relu', name='conv12_relu')(l)
	l = BatchNormalization(name='conv12_bn')(l)

	l = concatenate([l, l3, l5])

	"""
	Normal Convs
	"""

	l = Conv2D(192, (3, 3), padding='same', strides=(2, 2), name='conv13', trainable=trainable, kernel_regularizer=OrthLocalReg2D)(l)
	l = Activation('relu', name='conv13_relu')(l)
	l = BatchNormalization(name='conv13_bn')(l)

	l = SeparableConv2D(192, (3, 3), padding='same', name='conv14', trainable=trainable, depthwise_regularizer=OrthLocalRegSep2D)(l)
	l = Activation('relu', name='conv14_relu')(l)
	l_res_h = BatchNormalization(name='conv14_bn')(l)

	# l5 = SeparableConv2D(192, (5, 5), padding='same', name='conv14_5', trainable=trainable, depthwise_regularizer=OrthLocalRegSep2D)(l)
	# l5 = Activation('relu', name='conv14_5_relu')(l5)
	# l5 = BatchNormalization(name='conv14_5_bn')(l5)

	l = SeparableConv2D(192, (3, 3), padding='same', name='conv21', trainable=trainable, depthwise_regularizer=OrthLocalRegSep2D)(l_res_h)
	l = Activation('relu', name='conv21_relu')(l)
	l = BatchNormalization(name='conv21_bn')(l)

	# l = SeparableConv2D(192, (3, 3), padding='same', name='conv22', trainable=trainable, depthwise_regularizer=OrthLocalRegSep2D)(l)
	# l = Activation('relu', name='conv22_relu')(l)
	# l = BatchNormalization(name='conv22_bn')(l)
	l = Add(name='residual_21_23')([l_res_h, l])

	# l = concatenate([l, l5])

	# l5 = SeparableConv2D(384, (5, 5), padding='same', strides=(2, 2), name='conv23_5', trainable=trainable, depthwise_regularizer=OrthLocalRegSep2D)(l)
	# l5 = Activation('relu', name='conv23_5_relu')(l5)
	# l5 = BatchNormalization(name='conv23_5_bn')(l5)

	l = SeparableConv2D(384, (3, 3), padding='same', strides=(2, 2), name='conv23', trainable=trainable, depthwise_regularizer=OrthLocalRegSep2D)(l)
	l = Activation('relu', name='conv23_relu')(l)
	l = BatchNormalization(name='conv23_bn')(l)

	l = SeparableConv2D(384, (3, 3), padding='same', name='conv31', trainable=trainable, depthwise_regularizer=OrthLocalRegSep2D)(l)
	l = Activation('relu', name='conv31_relu')(l)
	l_res_h = BatchNormalization(name='conv31_bn')(l)

	l = SeparableConv2D(384, (3, 3), padding='same', name='conv32', trainable=trainable, depthwise_regularizer=OrthLocalRegSep2D)(l_res_h)
	l = Activation('relu', name='conv32_relu')(l)
	l = BatchNormalization(name='conv32_bn')(l)

	# l = SeparableConv2D(384, (3, 3), padding='same', name='conv33', trainable=trainable, depthwise_regularizer=OrthLocalRegSep2D)(l)
	# l = Activation('relu', name='conv33_relu')(l)
	# l = BatchNormalization(name='conv33_bn')(l)
	# l = Add(name='residual_31_33')([l_res_h, l, l5])

	# l = concatenate([l, l5])

	l_offset = ConvOffset2D(384, name='conv33_offset', kernel_regularizer=OrthLocalReg2D)(l, use_resam=True)
	l = Conv2D(512, (3, 3), padding='same', name='conv41', trainable=trainable, kernel_regularizer=OrthLocalReg2D)(l_offset)
	l = Activation('relu', name='conv41_relu')(l)
	l = BatchNormalization(name='conv41_bn')(l)

	l = SeparableConv2D(512, (3, 3), padding='same', strides=(2, 2), depth_multiplier=1, name='conv42', trainable=trainable, depthwise_regularizer=OrthLocalRegSep2D)(l)
	l = Activation('relu', name='conv42_relu')(l)
	l_res_h = BatchNormalization(name='conv42_bn')(l)

	# l_offset = ConvOffset2D(512, name='conv35_offset')(l)
	l = SeparableConv2D(512, (3, 3), padding='same', name='conv43', trainable=trainable, depthwise_regularizer=OrthLocalRegSep2D)(l_res_h)
	l = Activation('relu', name='conv43_relu')(l)
	l = BatchNormalization(name='conv43_bn')(l)

	# l = SeparableConv2D(512, (3, 3), padding='same', name='conv44', trainable=trainable, depthwise_regularizer=OrthLocalRegSep2D)(l)
	# l = Activation('relu', name='conv44_relu')(l)
	# l = BatchNormalization(name='conv44_bn')(l)
	l = Add(name='residual_42_44')([l_res_h, l])

	l = SeparableConv2D(1024, (3, 3), padding='same', strides=(2, 2), name='conv51', trainable=trainable, depthwise_regularizer=OrthLocalRegSep2D)(l)
	l = Activation('relu', name='conv51_relu')(l)
	l_res_h = BatchNormalization(name='conv51_bn')(l)

	l = SeparableConv2D(1024, (3, 3), padding='same', name='conv52', trainable=trainable, depthwise_regularizer=OrthLocalRegSep2D)(l_res_h)
	l = Activation('relu', name='conv52_relu')(l)
	l = BatchNormalization(name='conv52_bn')(l)

	# l = SeparableConv2D(1024, (3, 3), padding='same', name='conv53', trainable=trainable, depthwise_regularizer=OrthLocalRegSep2D)(l)
	# l = Activation('relu', name='conv53_relu')(l)
	# l = BatchNormalization(name='conv53_bn')(l)
	# l = Add(name='residual_51_53')([l_res_h, l])
	# out
	l = SpatialPyramidPooling([1, 2, 4], input_shape=[None, None, 1024])(l)

	# l = Flatten(name='flatten_maxpool')(l)
	l = Dense(768, name='fc1', trainable=trainable, kernel_regularizer=OrthLocalReg1D)(l)
	l = Activation('relu', name='fc1_relu')(l)

	l = Dense(256, name='fc2', trainable=trainable, kernel_regularizer=OrthLocalReg1D)(l)
	l = Activation('relu', name='fc2_relu')(l)

	outputs_1 = Dense(class_num, name='fc3', trainable=trainable)(l)
	outputs_1 = Activation('softmax', name='out')(outputs_1)

	outputs_2 = Dense(2, name='fc3_super', trainable=trainable)(l)
	outputs_2 = Activation('softmax')(outputs_2)

	# outputs = concatenate([outputs_1, outputs_2], name='out')
	outputs = [outputs_1, outputs_2]
	return inputs, outputs
