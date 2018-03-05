from __future__ import absolute_import, division

import keras as K
from keras.layers import (
	Input,
	Conv2D,
	SeparableConv2D,
	Activation,
	GlobalAvgPool2D,
	Dense,
	BatchNormalization,
	MaxPooling2D,
	AveragePooling2D,
	GlobalAveragePooling2D,
	GlobalMaxPooling2D,
	Flatten,
	Conv2D,
	Embedding,
	Lambda,
	Dropout,
	LocallyConnected2D,
	UpSampling2D,
)
from keras.layers.merge import concatenate, Add
from keras.models import Model
from keras import regularizers

from deform_conv.layers import *
from deform_conv.utils import make_parallel

from keras.initializers import Orthogonal, lecun_normal


def get_ewc_cnn(class_n, trainable=True):
	inputs = l = Input((28, 28, 1), name='input')

	conv_args = {
	 	'padding': 'same',
		'kernel_initializer': 'Orthogonal',
		'kernel_regularizer': regularizers.l2(0.01)
	}

	# l = ImageNorm()(l)

	# conv11
	# l2 = InvConv2D(32, (3, 3), strides=(2, 2), name='conv11_inv', **conv_args)(l)
	# l2 = Activation('relu', name='conv11_inv_relu')(l2)
	# l2 = BatchNormalization(name='conv11_inv_bn')(l2)
    #
	l = Conv2D(32, (3, 3), strides=(2, 2), name='conv11', **conv_args)(l)
	l = Activation('relu', name='conv11_relu')(l)
	l = BatchNormalization(name='conv11_bn')(l)
    #
	# l = concatenate([l, l2])
    #
	l = Conv2D(64, (1, 1), name='conv11_12', **conv_args)(l)
	l = Activation('relu', name='conv11_12_relu')(l)
	l = BatchNormalization(name='conv11_12_bn')(l)
    #
	# # conv12
	# l2 = InvConv2D(64, (3, 3), strides=(2, 2), name='conv13', **conv_args)(l)
	# l2 = Activation('relu', name='conv13_relu')(l2)
	# l2 = BatchNormalization(name='conv13_bn')(l2)
    #
	l = Conv2D(64, (3, 3), strides=(2, 2), name='conv12', **conv_args)(l)
	l = Activation('relu', name='conv12_relu')(l)
	l = BatchNormalization(name='conv12_bn')(l)
    #
	# l = concatenate([l, l2])
    #
	l = Conv2D(64, (1, 1), name='conv13_21', **conv_args)(l)
	l = Activation('relu', name='conv13_21_relu')(l)
	l = BatchNormalization(name='conv13_21_bn')(l)
    #
	# # conv22
	# l2 = InvConv2D(128, (3, 3), name='conv22', **conv_args)(l)
	# l2 = Activation('relu', name='conv22_relu')(l2)
	# l2 = BatchNormalization(name='conv22_bn')(l2)
    #
	# # conv21
	l = Conv2D(128, (3, 3), name='conv21', **conv_args)(l)
	l = Activation('relu', name='conv21_relu')(l)
	l = BatchNormalization(name='conv21_bn')(l)

	# l = concatenate([l, l2])
    #
	l = Conv2D(128, (1, 1), name='conv22_23', **conv_args)(l)
	l = Activation('relu', name='conv22_23_relu')(l)
	l = BatchNormalization(name='conv22_23_bn')(l)
    #
	# l = Conv2D(128, (3, 3), name='conv23', **conv_args)(l)
	# l = Activation('relu', name='conv23_relu')(l)
	# l = BatchNormalization(name='conv23_bn')(l)
    #
	# l = Conv2D(256, (3, 3), strides=(2, 2), name='conv31', **conv_args)(l)
	# l = Activation('relu', name='conv31_relu')(l)
	# l = BatchNormalization(name='conv31_bn')(l)
    #
	# l = Conv2D(256, (3, 3), name='conv32', **conv_args)(l)
	# l = Activation('relu', name='conv32_relu')(l)
	# l = BatchNormalization(name='conv32_bn')(l)
    #
	# l = Conv2D(512, (3, 3), strides=(2, 2), name='conv33', **conv_args)(l)
	# l = Activation('relu', name='conv33_relu')(l)
	# l = BatchNormalization(name='conv33_bn')(l)

	# out
	l = Flatten(name='flatten_maxpool')(l)
	l = Dropout(0.5)(l)
	# l = Dense(100, name='fc3')(l)
	# l = Activation('relu')(l)

    #
	# l = Dense(100, name='fc2')(l)
	# l = Activation('relu')(l)

	# l = GlobalAvgPool2D(name='avg_pool')(l)
	l = Dense(class_n, name='fc1')(l)
	outputs = l = Activation('softmax', name='out')(l)

	return inputs, outputs


def get_large_deform_cnn(class_num, trainable=False, dropout_sample=False):
	init = Orthogonal(gain=1.0, seed=None)

	inputs = l = Input((None, None, 3), name='input')

	norm_input = ImageNorm()(inputs)
	# norm_input = inputs

	# conv11
	l = Conv2D(32, (3, 3), strides=(2, 2), padding='same', name='conv11', trainable=trainable, kernel_initializer=init, kernel_regularizer=OrthLocalReg2D)(norm_input)
	l = Activation('relu', name='conv11_relu')(l)
	l = BatchNormalization(name='conv11_bn', center=False, scale=False)(l)

	l2 = InvConv2D(32, (3, 3), strides=(2, 2), padding='same', name='inv_conv11', trainable=trainable, kernel_initializer=init, kernel_regularizer=OrthLocalReg2D)(norm_input)
	l2 = Activation('relu', name='inv_conv11_relu')(l2)
	l2 = BatchNormalization(name='inv_conv11_bn', center=False, scale=False)(l2)


	# max_xor = ReduceMax2D()([l, l2])
	l = concatenate([l, l2])

	# conv12
	# l_offset = ConvOffset2D(32, name='conv12_offset')(l)

	l = Conv2D(128, (3, 3), padding='same', strides=(2, 2), name='pool11_12', trainable=trainable, kernel_initializer=init, kernel_regularizer=OrthLocalReg2D)(l)
	l = Activation('relu', name='pool11_12_relu')(l)
	l = BatchNormalization(name='pool11_12_bn', center=False, scale=False)(l)

	l5 = Conv2D(64, (5, 5), padding='same', name='conv12_5', trainable=trainable, kernel_initializer=init, kernel_regularizer=OrthLocalReg2D)(l)
	l5 = Activation('relu', name='conv12_5_relu')(l5)
	l5 = BatchNormalization(name='conv12_5_bn', center=False, scale=False)(l5)

	l = Conv2D(64, (3, 3), padding='same', name='conv12_3', trainable=trainable, kernel_initializer=init, kernel_regularizer=OrthLocalReg2D)(l)
	l = Activation('relu', name='conv12_3_relu')(l)
	l = BatchNormalization(name='conv12_3_bn', center=False, scale=False)(l)

	l = concatenate([l, l5])

	l = Conv2D(128, (3, 3), padding='same', name='conv13', trainable=trainable, kernel_initializer=init, kernel_regularizer=OrthLocalReg2D)(l)
	l = Activation('relu', name='conv13_relu')(l)
	l = BatchNormalization(name='conv13_bn', center=False, scale=False)(l)

	l = Conv2D(128, (3, 3), padding='same', strides=(2, 2), name='conv14', trainable=trainable, kernel_initializer=init, kernel_regularizer=OrthLocalReg2D)(l)
	l = Activation('relu', name='conv14_relu')(l)
	l = BatchNormalization(name='conv14_bn', center=False, scale=False)(l)


	l = Conv2D(256, (3, 3), padding='same', strides=(2, 2), name='conv31', trainable=trainable, kernel_initializer=init, kernel_regularizer=OrthLocalReg2D)(l)
	l = Activation('relu', name='conv31_relu')(l)
	l31 = BatchNormalization(name='conv31_bn', center=False, scale=False)(l)

	l = Conv2D(256, (1, 1), padding='same', name='conv32', trainable=trainable, kernel_initializer=init, kernel_regularizer=OrthLocalReg2D)(l31)
	l = Activation('relu', name='conv32_relu')(l)
	l = BatchNormalization(name='conv32_bn', center=False, scale=False)(l)

	l = auxinliary_branch = Conv2D(256, (3, 3), padding='same', name='conv33', trainable=trainable, kernel_initializer=init, kernel_regularizer=OrthLocalReg2D)(l)
	l = Activation('relu', name='conv33_relu')(l)
	l = BatchNormalization(name='conv33_bn', center=False, scale=False)(l)
	l = Add(name='residual_31_32')([l31, l])


	l_offset = ConvOffset2D(256, name='conv33_offset')(l)
	l = Conv2D(512, (3, 3), padding='same', name='conv41', trainable=trainable, kernel_initializer=init, kernel_regularizer=OrthLocalReg2D)(l_offset)
	l = Activation('relu', name='conv41_relu')(l)
	l = BatchNormalization(name='conv41_bn', center=False, scale=False)(l)

	l = Conv2D(512, (3, 3), padding='same', strides=(2, 2), name='conv42', trainable=trainable, kernel_initializer=init, kernel_regularizer=OrthLocalReg2D)(l)
	l = Activation('relu', name='conv42_relu')(l)
	l = BatchNormalization(name='conv42_bn', center=False, scale=False)(l)

	# l_offset = ConvOffset2D(512, name='conv35_offset')(l)
	l = Conv2D(512, (3, 3), padding='same', name='conv43', trainable=trainable, kernel_initializer=init, kernel_regularizer=OrthLocalReg2D)(l)
	l = Activation('relu', name='conv43_relu')(l)
	l = BatchNormalization(name='conv43_bn', center=False, scale=False)(l)

	l = Conv2D(1024, (3, 3), padding='same', strides=(2, 2), name='conv51', trainable=trainable, kernel_initializer=init, kernel_regularizer=OrthLocalReg2D)(l)
	l = Activation('relu', name='conv51_relu')(l)
	l = BatchNormalization(name='conv51_bn', center=False, scale=False)(l)

	l = Conv2D(1024, (3, 3), padding='same', name='conv52', trainable=trainable, kernel_initializer=init, kernel_regularizer=OrthLocalReg2D)(l)
	l = Activation('relu', name='conv52_relu')(l)
	l = BatchNormalization(name='conv52_bn', center=False, scale=False)(l)

	l = Conv2D(1024, (3, 3), padding='same', name='conv53', trainable=trainable, kernel_initializer=init, kernel_regularizer=OrthLocalReg2D)(l)
	l = Activation('relu', name='conv53_relu')(l)
	l = BatchNormalization(name='conv53_bn', center=False, scale=False)(l)

	# out
	l = GlobalAvgPool2D(name='avg_pool')(l)
	# l = MaxPooling2D(name='max_pool_final')(l)
	# l = Flatten(name='flatten_maxpool')(l)
	if dropout_sample:
		l = Dropout(0.5)(l)
	l = Dense(512, name='fc1', trainable=trainable, kernel_initializer=init)(l)
	l = Activation('relu', name='fc1_relu')(l)
	l = BatchNormalization()(l)

	if dropout_sample:
		l = Dropout(0.5)(l)
	l = Dense(256, name='fc2', trainable=trainable, kernel_initializer=init)(l)
	l = Activation('relu', name='fc2_relu')(l)
	l = BatchNormalization()(l)

	l = Dense(class_num, name='fc3', trainable=trainable)(l)
	outputs = l = Activation('softmax', name='out')(l)

	return inputs, outputs


def deform_center_cnn(class_num, trainable=False, GPU=1):
	conv_args = {
		'trainable': trainable,
		'kernel_initializer': Orthogonal(gain=1.0, seed=None),
		'kernel_regularizer': OrthLocalReg2D,
		'padding': 'same'
	}

	sep_conv_args = {
		'trainable': trainable,
		'kernel_initializer': Orthogonal(gain=1.0, seed=None),
		'kernel_regularizer': OrthLocalRegSep2D,
		'padding': 'same'
	}


	inputs = l = Input((None, None, 3), name='input')
	input_target = Input((1,), name='input_target')

	# norm_input = RGB2Gray()(inputs)
	norm_input = ImageNorm()(inputs)
	# norm_input = inputs
	stem_stride = (2, 2)
	# conv11
	l = Conv2D(32, (3, 3), strides=stem_stride, name='conv11', **conv_args)(norm_input)
	l = Activation('relu', name='conv11_relu')(l)
	l = BatchNormalization(name='conv11_bn')(l)

	l2 = InvConv2D(32, (3, 3), strides=stem_stride, name='inv_conv11', **conv_args)(norm_input)
	l2 = Activation('relu', name='inv_conv11_relu')(l2)
	l2 = BatchNormalization(name='inv_conv11_bn')(l2)

	l = concatenate([l, l2])

	l5 = SeparableConv2D(64, (5, 5), strides=(2, 2), name='conv5_11_12', **sep_conv_args)(l)
	l5 = Activation('relu', name='conv5_11_12_relu')(l5)
	l5 = BatchNormalization(name='conv5_11_12_bn')(l5)

	l3 = SeparableConv2D(64, (3, 3), strides=(2, 2), name='conv3_11_12', **sep_conv_args)(l)
	l3 = Activation('relu', name='conv3_11_12_relu')(l3)
	l3 = BatchNormalization(name='conv3_11_12_bn')(l3)

	l = concatenate([l3, l5])

	l = SeparableConv2D(128, (3, 3), name='conv12_1', **sep_conv_args)(l)
	l = Activation('relu', name='conv12_1_relu')(l)
	l = BatchNormalization(name='conv12_1_bn')(l)

	l = SeparableConv2D(128, (1, 1), name='conv12_2', **sep_conv_args)(l)
	l = Activation('relu', name='conv12_2_relu')(l)
	l = BatchNormalization(name='conv12_2_bn')(l)

	l = SeparableConv2D(128, (3, 3), name='conv13', **sep_conv_args)(l)
	l = Activation('relu', name='conv13_relu')(l)
	l = BatchNormalization(name='conv13_bn')(l)

	l = SeparableConv2D(256, (3, 3), strides=(2, 2), name='conv14', **sep_conv_args)(l)
	l = Activation('relu', name='conv14_relu')(l)
	l = l14 = BatchNormalization(name='conv14_bn')(l)

	l = SeparableConv2D(256, (3, 3), name='conv21', **sep_conv_args)(l)
	l = Activation('relu', name='conv21_relu')(l)
	l = BatchNormalization(name='conv21_bn')(l)

	l = SeparableConv2D(256, (3, 3), name='conv22', **sep_conv_args)(l)
	l = Activation('relu', name='conv22_relu')(l)
	l = BatchNormalization(name='conv22_bn')(l)

	l = Add(name='residual_14_22')([l14, l])

	# conv22
	# l_offset = ConvOffset2D(192, name='conv32_offset')(l)
	# l = Conv2D(256, (3, 3), strides=(2, 2), name='conv23', **conv_args)(l)
	# l = Activation('relu', name='conv23_relu')(l)
	# l = l23 = BatchNormalization(name='conv23_bn')(l)
    #
	# l = Conv2D(256, (3, 3), name='conv31', **conv_args)(l)
	# l = Activation('relu', name='conv31_relu')(l)
	# l = BatchNormalization(name='conv31_bn')(l)

	# l = Conv2D(256, (1, 1), name='conv32', **conv_args)(l31)
	# l = Activation('relu', name='conv32_relu')(l)
	# l = BatchNormalization(name='conv32_bn')(l)

	# l = Conv2D(256, (3, 3), name='conv33', **conv_args)(l)
	# l = Activation('relu', name='conv33_relu')(l)
	# l = BatchNormalization(name='conv33_bn')(l)
	# l = Add(name='residual_23_33')([l23, l])


	l_offset = ConvOffset2D(256, name='conv33_offset')(l)
	l = SeparableConv2D(512, (3, 3), strides=(2, 2), name='conv41', **sep_conv_args)(l_offset)
	l = Activation('relu', name='conv41_relu')(l)
	l = BatchNormalization(name='conv41_bn')(l)

	l = SeparableConv2D(512, (3, 3), name='conv42', **sep_conv_args)(l)
	l = Activation('relu', name='conv42_relu')(l)
	l = BatchNormalization(name='conv42_bn')(l)

	# l_offset = ConvOffset2D(512, name='conv35_offset')(l)
	l = SeparableConv2D(512, (3, 3), name='conv43', **sep_conv_args)(l)
	l = Activation('relu', name='conv43_relu')(l)
	l = BatchNormalization(name='conv43_bn')(l)

	# l = LocallyConnected2D(512, (3, 3), name='conv44', padding='valid')(l)
	# l = Activation('relu', name='conv44_relu')(l)
	# l = BatchNormalization(name='conv44_bn')(l)

	l = SeparableConv2D(1024, (3, 3), strides=(2, 2), name='conv51', **sep_conv_args)(l)
	l = Activation('relu', name='conv51_relu')(l)
	l = BatchNormalization(name='conv51_bn')(l)

	# l_offset = ConvOffset2D(1024, name='conv35_offset')(l)
	l = SeparableConv2D(1024, (3, 3), name='conv52', **sep_conv_args)(l)
	l = Activation('relu', name='conv52_relu')(l)
	l = BatchNormalization(name='conv52_bn')(l)

	l = SeparableConv2D(1024, (3, 3), name='conv53', **sep_conv_args)(l)
	l = Activation('relu', name='conv53_relu')(l)
	l = BatchNormalization(name='conv53_bn')(l)

	# out
	l = GlobalAvgPool2D(name='avg_pool')(l)
	# l = MaxPooling2D(name='max_pool_final')(l)
	# l = Flatten(name='flatten_maxpool')(l)

	l = Dense(512, name='fc1', trainable=trainable)(l)
	l = Activation('relu', name='fc1_relu')(l)

	l = Dense(256, name='fc2', trainable=trainable)(l)
	l = feature = Activation('relu', name='fc2_relu')(l)

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


def get_DCNN(class_num, trainable=False):
	inputs = l = Input((None, None, 3), name='input')

	"""
	Block 1
	"""
	l = Conv2D(32, (3, 3), padding='same', strides=(2, 2), name='conv11', trainable=trainable, kernel_regularizer=OrthLocalReg2D)(l)
	l = Activation('relu', name='conv11_relu')(l)
	l = BatchNormalization(name='conv11_bn')(l)

	l2 = InvConv2D(32, (3, 3), padding='same', strides=(2, 2), name='inv_conv11', trainable=trainable, kernel_regularizer=OrthLocalReg2D)(inputs)
	l2 = Activation('relu', name='inv_conv11_relu')(l2)
	l2 = BatchNormalization(name='inv_conv11_bn')(l2)

	l3 = Conv2D(32, (7, 7), padding='same', strides=(2, 2), name='conv11_2', trainable=trainable, kernel_regularizer=OrthLocalReg2D)(inputs)
	l3 = Activation('relu', name='conv11_2_relu')(l3)
	l3 = BatchNormalization(name='conv11_2_bn')(l3)

	l5 = InvConv2D(32, (7, 7), padding='same', strides=(2, 2), name='conv11_3', trainable=trainable, kernel_regularizer=OrthLocalReg2D)(inputs)
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
	anx_output = l = BatchNormalization(name='conv41_bn')(l)

	l = SeparableConv2D(512, (3, 3), padding='same', strides=(2, 2), depth_multiplier=1, name='conv42', trainable=trainable, depthwise_regularizer=OrthLocalRegSep2D)(l)
	l = Activation('relu', name='conv42_relu')(l)
	l_res_h = BatchNormalization(name='conv42_bn')(l)

	# l = CapsuleRouting(512, 512, 169, 169, name='cp1', reshape_cnn=True,)(l_res_h)
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

	# l = CapsuleRouting(1024, 1024, 49, 49, name='cp2', reshape_cnn=True,)(l)
	l = SeparableConv2D(1024, (3, 3), padding='same', name='conv53', trainable=trainable, depthwise_regularizer=OrthLocalRegSep2D)(l)
	l = Activation('relu', name='conv53_relu')(l)
	l = BatchNormalization(name='conv53_bn')(l)
	l = Add(name='residual_51_53')([l_res_h, l])
	# out
	# l = SpatialPyramidPooling([1, 2, 4], input_shape=[None, None, 1024])(l)
	l = GlobalAveragePooling2D()(l)

	# l = Flatten(name='flatten_maxpool')(l)
	l = Dense(512, name='fc1', trainable=trainable, kernel_regularizer=OrthLocalReg1D)(l)
	l = Activation('relu', name='fc1_relu')(l)

	l2 = Dense(256, name='fc2_2', trainable=trainable)(l)
	l2 = Activation('relu', name='fc2_2_relu')(l2)

	l = Dense(256, name='fc2', trainable=trainable)(l)
	l = Activation('relu', name='fc2_relu')(l)

    #
	# outputs_1 = Dense(class_num, name='fc3', trainable=trainable)(l)
	# outputs_1 = Activation('softmax', name='out')(outputs_1)
    #
	# outputs_2 = Dense(2, name='fc3_super', trainable=trainable)(l)
	# outputs_2 = Activation('softmax')(outputs_2)

	l2 = Dense(1, name='fc3_2', trainable=trainable)(l2)
	l2 = Activation('sigmoid', name='gate')(l2)

	l = Dense(class_num, name='fc3', trainable=trainable)(l)
	l = Activation('softmax', name='out')(l)

	# outputsgated_softmax(class_num, name='gate')()

	# outputs = concatenate([outputs_1, outputs_2], name='out')
	# outputs = [outputs_1, outputs_2]
	outputs = [l, l2]
	return inputs, outputs


def get_test_cnn(class_n, trainable=True, dropout_sample=False):

	conv_args = {
	 	'padding': 'same',
		'kernel_initializer': 'Orthogonal',
		'kernel_regularizer': OrthLocalReg2D
	}

	def block(X, F, O, L, block_name):
		x_d = x_u = X

		for l in range(L):
			x_d = Conv2D(F*2**l, (3, 3), strides=(2, 2), name='%s_d%d' % (block_name, l), **conv_args)(x_d)
			x_d = Activation('relu', name='%s_d%d_relu' % (block_name, l))(x_d)
			x_d = BatchNormalization(name='%s_d%d_bn' % (block_name, l))(x_d)

			x_u = Conv2D(F*2**((l+1)%2), (3, 3), name='%s_u%d' % (block_name, l), **conv_args)(x_u)
			x_u = Activation('relu', name='%s_u%d_relu' % (block_name, l))(x_u)
			x_u = BatchNormalization(name='%s_u%d_bn' % (block_name, l))(x_u)

		for l in range(L):
			x_d = UpSampling2D(size=(2, 2))(x_d)

		x_d_u = concatenate([x_d, x_u])
		x_d_u = Conv2D(O, (1, 1), name='%s_redu' % block_name, **conv_args)(x_d_u)
		x_d_u = Activation('relu', name='%s_redu_relu' % block_name)(x_d_u)
		x_d_u = BatchNormalization(name='%s_redu_bn' % block_name)(x_d_u)

		if F == O:
			x_d_u = Add()([x_d_u, X])
		return x_d_u

	inputs = l = Input((224, 224, 3), name='input')

	l = ImageNorm()(l)
# 100 50 25 13 6
	l = Conv2D(32, (3, 3), strides=(2, 2), name='conv11', **conv_args)(l)
	l = Activation('relu', name='conv11_relu')(l)
	l = BatchNormalization(name='conv11_bn')(l)

	l = block(l, 32, 64, 3, 'L1_3')

	l = Conv2D(128, (3, 3), strides=(2, 2), name='conv21', **conv_args)(l)
	l = Activation('relu', name='conv21_relu')(l)
	l = BatchNormalization(name='conv21_bn')(l)

	l = block(l, 128, 128, 3, 'L2_3')

	l = ConvOffset2D(128, name='conv31_offset')(l, use_resam=trainable)
	l = Conv2D(256, (3, 3), strides=(2, 2), name='conv31', **conv_args)(l)
	l = Activation('relu', name='conv31_relu')(l)
	l = BatchNormalization(name='conv31_bn')(l)

	l = block(l, 256, 256, 2, 'L3_3')

	l = Conv2D(512, (3, 3), strides=(2, 2), name='conv41', **conv_args)(l)
	l = Activation('relu', name='conv41_relu')(l)
	l = BatchNormalization(name='conv41_bn')(l)

	l = GlobalAveragePooling2D()(l)

	l = Dense(128, name='fc1')(l)
	l = Activation('relu')(l)
	l = BatchNormalization(name='fc1_bn')(l)

	# l = GlobalAvgPool2D(name='avg_pool')(l)
	l = Dense(class_n, name='fc2')(l)
	outputs = l = Activation('softmax', name='out')(l)

	return inputs, outputs
