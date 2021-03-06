from __future__ import absolute_import, division


from keras.layers import Input, Conv2D, SeparableConv2D, Activation, GlobalAvgPool2D, Dense, BatchNormalization, MaxPooling2D, Flatten, Conv2D, Lambda
from keras.layers.merge import concatenate, Add
from deform_conv.layers import *


def get_cnn():
    inputs = l = Input((28, 28, 1), name='input')

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
    l = Dense(10, name='fc1')(l)
    outputs = l = Activation('softmax', name='out')(l)

    return inputs, outputs


def get_large_deform_cnn(class_num, trainable=False, to_TF=False):
    inputs = l = Input((200, 200, 3), name='input')

    # if to_TF:
    #     l = Lambda(lambda x: 255 * x)(inputs)

    # conv11
    l2 = InvConv2D(32, (3, 3), padding='same', name='inv_conv11', trainable=trainable, kernel_regularizer=OrthLocalReg2D)(l)
    l2 = Activation('relu', name='inv_conv11_relu')(l2)
    l2 = BatchNormalization(name='inv_conv11_bn')(l2)

    l3 = Conv2D(32, (3, 1), padding='same', name='conv11_2', trainable=trainable, kernel_regularizer=OrthLocalReg2D)(l)
    l3 = Activation('relu', name='conv11_2_relu')(l3)
    l3 = BatchNormalization(name='conv11_2_bn')(l3)

    l5 = Conv2D(32, (1, 3), padding='same', name='conv11_3', trainable=trainable, kernel_regularizer=OrthLocalReg2D)(l)
    l5 = Activation('relu', name='conv11_3_relu')(l5)
    l5 = BatchNormalization(name='conv11_3_bn')(l5)

    l = Conv2D(32, (3, 3), padding='same', name='conv11', trainable=trainable, kernel_regularizer=OrthLocalReg2D)(l)
    l = Activation('relu', name='conv11_relu')(l)
    l = BatchNormalization(name='conv11_bn')(l)

    l = concatenate([l, l2, l3, l5])

    # conv12
    # l_offset = ConvOffset2D(32, name='conv12_offset')(l)

    l = Conv2D(128, (3, 3), padding='same', strides=(2, 2), name='pool11_12', trainable=trainable, kernel_regularizer=OrthLocalReg2D)(l)
    l = Activation('relu', name='pool11_12_relu')(l)
    # l = BatchNormalization(name='conv12_bn')(l)


    l3 = Conv2D(32, (3, 1), padding='same', name='conv12_2', trainable=trainable, kernel_regularizer=OrthLocalReg2D)(l)
    l3 = Activation('relu', name='conv12_2_relu')(l3)
    l3 = BatchNormalization(name='conv12_2_bn')(l3)

    l5 = Conv2D(32, (1, 3), padding='same', name='conv12_3', trainable=trainable, kernel_regularizer=OrthLocalReg2D)(l)
    l5 = Activation('relu', name='conv12_3_relu')(l5)
    l5 = BatchNormalization(name='conv12_3_bn')(l5)

    l = Conv2D(128, (3, 3), padding='same', name='conv12', trainable=trainable, kernel_regularizer=OrthLocalReg2D)(l)
    l = Activation('relu', name='conv12_relu')(l)
    l = BatchNormalization(name='conv12_bn')(l)

    l = concatenate([l, l3, l5])

    l = Conv2D(128, (3, 3), padding='same', name='conv13', trainable=trainable, kernel_regularizer=OrthLocalReg2D)(l)
    l = Activation('relu', name='conv13_relu')(l)
    l = BatchNormalization(name='conv13_bn')(l)

    l = Conv2D(128, (3, 3), padding='same', strides=(2, 2), name='conv14', trainable=trainable, kernel_regularizer=OrthLocalReg2D)(l)
    l = Activation('relu', name='conv14_relu')(l)
    # l = BatchNormalization(name='conv14_bn')(l)

    l = Conv2D(192, (3, 3), padding='same', name='conv21', trainable=trainable, kernel_regularizer=OrthLocalReg2D)(l)
    l = Activation('relu', name='conv21_relu')(l)
    l = BatchNormalization(name='conv21_bn')(l)

    l = Conv2D(192, (3, 3), padding='same', name='conv22', trainable=trainable, kernel_regularizer=OrthLocalReg2D)(l)
    l = Activation('relu', name='conv22_relu')(l)
    l = BatchNormalization(name='conv22_bn')(l)

    # conv22
    # l_offset = ConvOffset2D(192, name='conv32_offset')(l)
    l = Conv2D(192, (3, 3), padding='same', strides=(2, 2), name='conv23', trainable=trainable, kernel_regularizer=OrthLocalReg2D)(l)
    l = Activation('relu', name='conv23_relu')(l)
    l = BatchNormalization(name='conv23_bn')(l)

    l = Conv2D(256, (3, 3), padding='same', name='conv31', trainable=trainable, kernel_regularizer=OrthLocalReg2D)(l)
    l = Activation('relu', name='conv31_relu')(l)
    l31 = BatchNormalization(name='conv31_bn')(l)

    l = Conv2D(256, (1, 1), padding='same', name='conv32', trainable=trainable, kernel_regularizer=OrthLocalReg2D)(l31)
    l = Activation('relu', name='conv32_relu')(l)
    l = BatchNormalization(name='conv32_bn')(l)

    l = Conv2D(256, (3, 3), padding='same', name='conv33', trainable=trainable, kernel_regularizer=OrthLocalReg2D)(l)
    l = Activation('relu', name='conv33_relu')(l)
    l = BatchNormalization(name='conv33_bn')(l)
    l = Add(name='residual_31_32')([l31, l])


    l_offset = ConvOffset2D(256, name='conv33_offset')(l)
    l = Conv2D(512, (3, 3), padding='same', name='conv41', trainable=trainable, kernel_regularizer=OrthLocalReg2D)(l_offset)
    l = Activation('relu', name='conv41_relu')(l)
    l = BatchNormalization(name='conv41_bn')(l)

    l = Conv2D(512, (3, 3), padding='same', strides=(2, 2), name='conv42', trainable=trainable, kernel_regularizer=OrthLocalReg2D)(l)
    l = Activation('relu', name='conv42_relu')(l)
    l = BatchNormalization(name='conv42_bn')(l)

    # l_offset = ConvOffset2D(512, name='conv35_offset')(l)
    l = Conv2D(512, (3, 3), padding='same', name='conv43', trainable=trainable, kernel_regularizer=OrthLocalReg2D)(l)
    l = Activation('relu', name='conv43_relu')(l)
    l = BatchNormalization(name='conv43_bn')(l)

    # out
    # l = GlobalAvgPool2D(name='avg_pool')(l)
    l = MaxPooling2D(name='max_pool_final')(l)
    l = Flatten(name='flatten_maxpool')(l)
    l = Dense(768, name='fc1', trainable=trainable)(l)
    l = Activation('relu', name='fc1_relu')(l)

    l = Dense(256, name='fc2', trainable=trainable)(l)
    l = Activation('relu', name='fc2_relu')(l)

    l = Dense(class_num, name='fc3', trainable=trainable)(l)
    outputs = l = Activation('softmax', name='out')(l)

    return inputs, outputs


def get_large_deform_inv_cnn(class_num, trainable=False):
    inputs = l = Input((200, 200, 3), name='input')

    # conv11
    l = Conv2D(32, (3, 3), padding='same', name='conv11', trainable=trainable)(l)
    l = Activation('relu', name='conv11_relu')(l)
    l = BatchNormalization(name='conv11_bn')(l)

    l2 = InvConv2D(32, (3, 3), padding='same', name='inv_conv11', trainable=trainable)(inputs)
    l2 = Activation('relu', name='inv_conv11_relu')(l2)
    l2 = BatchNormalization(name='inv_conv11_bn')(l2)

    l3 = Conv2D(32, (3, 5), padding='same', name='conv11_2', trainable=trainable)(inputs)
    l3 = Activation('relu', name='conv11_2_relu')(l3)
    l3 = BatchNormalization(name='conv11_2_bn')(l3)

    l5 = Conv2D(32, (5, 3), padding='same', name='conv11_3', trainable=trainable)(inputs)
    l5 = Activation('relu', name='conv11_3_relu')(l5)
    l5 = BatchNormalization(name='conv11_3_bn')(l5)

    l = concatenate([l, l2, l3, l5])

    # conv12
    # l_offset = ConvOffset2D(32, name='conv12_offset')(l)

    l = Conv2D(128, (3, 3), padding='same', strides=(2, 2), name='pool11_12', trainable=trainable)(l)
    l = Activation('relu', name='pool11_12_relu')(l)
    # l = BatchNormalization(name='conv12_bn')(l)


    l3 = Conv2D(32, (3, 5), padding='same', name='conv12_2', trainable=trainable)(l)
    l3 = Activation('relu', name='conv12_2_relu')(l3)
    l3 = BatchNormalization(name='conv12_2_bn')(l3)

    l5 = Conv2D(32, (5, 3), padding='same', name='conv12_3', trainable=trainable)(l)
    l5 = Activation('relu', name='conv12_3_relu')(l5)
    l5 = BatchNormalization(name='conv12_3_bn')(l5)

    l = Conv2D(128, (3, 3), padding='same', name='conv12', trainable=trainable)(l)
    l = Activation('relu', name='conv12_relu')(l)
    l = BatchNormalization(name='conv12_bn')(l)
    l_block2 = l

    l = concatenate([l, l3, l5])

    l = Conv2D(128, (3, 3), padding='same', name='conv13', trainable=trainable)(l)
    l = Activation('relu', name='conv13_relu')(l)
    l = BatchNormalization(name='conv13_bn')(l)

    l = Conv2D(128, (3, 3), padding='same', strides=(2, 2), name='conv14', trainable=trainable)(l)
    l = Activation('relu', name='conv14_relu')(l)
    # l = BatchNormalization(name='conv14_bn')(l)

    l = Conv2D(192, (3, 3), padding='same', name='conv21', trainable=trainable)(l)
    l = Activation('relu', name='conv21_relu')(l)
    l = BatchNormalization(name='conv21_bn')(l)

    l = Conv2D(192, (3, 3), padding='same', name='conv22', trainable=trainable)(l)
    l = Activation('relu', name='conv22_relu')(l)
    l = BatchNormalization(name='conv22_bn')(l)

    # conv22
    # l_offset = ConvOffset2D(192, name='conv32_offset')(l)
    l = Conv2D(192, (3, 3), padding='same', strides=(2, 2), name='conv23', trainable=trainable)(l)
    l = Activation('relu', name='conv23_relu')(l)
    l = BatchNormalization(name='conv23_bn')(l)

    l24 = Conv2D(192, (7, 7), padding='same', strides=(4, 4), name='conv24', trainable=trainable)(l_block2)
    l24 = Activation('relu', name='conv24_relu')(l24)
    l24 = BatchNormalization(name='conv24_bn')(l24)

    l = concatenate([l ,l24])

    l = Conv2D(256, (3, 3), padding='same', name='conv31', trainable=trainable)(l)
    l = Activation('relu', name='conv31_relu')(l)
    l31 = BatchNormalization(name='conv31_bn')(l)

    l = Conv2D(256, (3, 3), padding='same', name='conv32', trainable=trainable)(l31)
    l = Activation('relu', name='conv32_relu')(l)
    l = BatchNormalization(name='conv32_bn')(l)

    l = Conv2D(256, (3, 3), padding='same', name='conv33', trainable=trainable)(l)
    l = Activation('relu', name='conv33_relu')(l)
    l = BatchNormalization(name='conv33_bn')(l)
    l = Add(name='residual_31_32')([l31, l])


    l_offset = ConvOffset2D(256, name='conv33_offset')(l)
    l = Conv2D(512, (3, 3), padding='same', name='conv41', trainable=trainable)(l_offset)
    l = Activation('relu', name='conv41_relu')(l)
    l = BatchNormalization(name='conv41_bn')(l)

    l = Conv2D(512, (3, 3), padding='same', strides=(2, 2), name='conv42', trainable=trainable)(l)
    l = Activation('relu', name='conv42_relu')(l)
    l = BatchNormalization(name='conv42_bn')(l)

    # l_offset = ConvOffset2D(512, name='conv35_offset')(l)
    l = Conv2D(512, (3, 3), padding='same', name='conv43', trainable=trainable)(l)
    l = Activation('relu', name='conv43_relu')(l)
    l = BatchNormalization(name='conv43_bn')(l)

    l = Conv2D(512, (3, 3), padding='same', name='conv44', trainable=trainable)(l)
    l = Activation('relu', name='conv44_relu')(l)
    l = BatchNormalization(name='conv44_bn')(l)

    # out
    # l = MaxPooling2D(name='max_pool_final')(l)
    # l = Flatten(name='flatten_maxpool')(l)

    l = SpatialPyramidPooling([1, 2, 4], input_shape=[None, None, 512])(l)
    l = Dense(768, name='fc1', trainable=trainable)(l)
    l = Activation('relu', name='fc1_relu')(l)

    l = Dense(256, name='fc2', trainable=trainable)(l)
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
