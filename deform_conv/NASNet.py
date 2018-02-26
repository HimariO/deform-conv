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
    ZeroPadding2D,
    Flatten,
    Embedding,
    Lambda,
    Cropping2D,
)
from keras.layers.merge import concatenate, Add
from keras.models import Model
from deform_conv.layers import *
from deform_conv.utils import make_parallel

from keras.initializers import Orthogonal, lecun_normal

def NASNet(class_num, trainable=True, img_size=200):
    def _adjust_block(p, ip, filters, weight_decay=5e-5, id=None):
        '''
        Adjusts the input `p` to match the shape of the `input`
        or situations where the output number of filters needs to
        be changed

        # Arguments:
            p: input tensor which needs to be modified
            ip: input tensor whose shape needs to be matched
            filters: number of output filters to be matched
            weight_decay: l2 regularization weight
            id: string id

        # Returns:
            an adjusted Keras tensor
        '''
        channel_dim = 1 if K.image_data_format() == 'channels_first' else -1
        img_dim = 2 if K.image_data_format() == 'channels_first' else -2

        with K.name_scope('adjust_block'):
            if p is None:
                p = ip

            elif p._keras_shape[img_dim] != ip._keras_shape[img_dim]:
                with K.name_scope('adjust_reduction_block_%s' % id):
                    p = Activation('relu', name='adjust_relu_1_%s' % id)(p)

                    p1 = AveragePooling2D((1, 1), strides=(2, 2), padding='valid', name='adjust_avg_pool_1_%s' % id)(p)
                    p1 = Conv2D(filters // 2, (1, 1), padding='same', use_bias=False, kernel_regularizer=OrthLocalReg2D,
                                name='adjust_conv_1_%s' % id, kernel_initializer='he_normal')(p1)

                    p2 = ZeroPadding2D(padding=((0, 1), (0, 1)))(p)
                    p2 = Cropping2D(cropping=((1, 0), (1, 0)))(p2)
                    p2 = AveragePooling2D((1, 1), strides=(2, 2), padding='valid', name='adjust_avg_pool_2_%s' % id)(p2)
                    p2 = Conv2D(filters // 2, (1, 1), padding='same', use_bias=False, kernel_regularizer=OrthLocalReg2D,
                                name='adjust_conv_2_%s' % id, kernel_initializer='he_normal')(p2)

                    p = concatenate([p1, p2], axis=channel_dim)
                    p = BatchNormalization(name='adjust_bn_%s' % id)(p)

            elif p._keras_shape[channel_dim] != filters:
                with K.name_scope('adjust_projection_block_%s' % id):
                    p = Activation('relu')(p)
                    p = Conv2D(filters, (1, 1), strides=(1, 1), padding='same', name='adjust_conv_projection_%s' % id,
                               use_bias=False, kernel_regularizer=OrthLocalReg2D, kernel_initializer='he_normal')(p)
                    p = BatchNormalization(name='adjust_bn_%s' % id)(p)
        return p


    def NormalCell(hi, h_i1sub, filter_i, filter_o, stride=1, name="NAS", use_deform=False, is_tail=False):
        """
        adjust feature size & channel size
        """

        hi = Conv2D(filter_o, (1, 1), padding='same', name='%s_hi_align' % name, trainable=trainable, kernel_regularizer=OrthLocalReg2D)(hi)
        hi = Activation('relu', name='%s_hi_align_relu' % name)(hi)
        hi = BatchNormalization(name='%s_hi_align_bn' % name)(hi)

        if use_deform:
            hi = SepConvOffset2D(filter_o, name='%s_deform_conv' % name)(hi, use_resam=True)

        h_i1sub = _adjust_block(h_i1sub, hi, filter_o, 0, name)

        """
        SubLayer
        """

        l = SeparableConv2D(filter_o, (3, 3), padding='same', name='%s_sep3x3_1' % name, trainable=trainable, depthwise_regularizer=OrthLocalRegSep2D)(hi)
        l = Activation('relu', name='%s_sep3x3_1_relu' % name)(l)
        l = BatchNormalization(name='%s_sep3x3_1_bn' % name)(l)

        add1 = Add()([l, hi])

        l = SeparableConv2D(filter_o, (5, 5), padding='same', name='%s_sep5x5_1' % name, trainable=trainable, depthwise_regularizer=OrthLocalRegSep2D)(hi)
        l = Activation('relu', name='%s_sep5x5_1_relu' % name)(l)
        l = BatchNormalization(name='%s_sep5x5_1_bn' % name)(l)

        if h_i1sub is not None:
            l_1sub = SeparableConv2D(filter_o, (3, 3), padding='same', name='%s_sep3x3_sub_1' % name, trainable=trainable, depthwise_regularizer=OrthLocalRegSep2D)(h_i1sub)
            l_1sub = Activation('relu', name='%s_sep3x3_sub_1_relu' % name)(l_1sub)
            l_1sub = BatchNormalization(name='%s_sep3x3_sub_1_bn' % name)(l_1sub)
            add2 = Add()([l, l_1sub])
        else:
            add2 = l

        l = SeparableConv2D(filter_o, (3, 3), padding='same', name='%s_sep3x3_2' % name, trainable=trainable, depthwise_regularizer=OrthLocalRegSep2D)(hi)
        l = Activation('relu', name='%s_sep3x3_2_relu' % name)(l)
        l = BatchNormalization(name='%s_sep3x3_2_bn' % name)(l)

        if h_i1sub is not None:
            l_1sub = SeparableConv2D(filter_o, (5, 5), padding='same', name='%s_sep5x5_sub_1' % name, trainable=trainable, depthwise_regularizer=OrthLocalRegSep2D)(h_i1sub)
            l_1sub = Activation('relu', name='%s_sep5x5_sub_1_relu' % name)(l_1sub)
            l_1sub = BatchNormalization(name='%s_sep5x5_sub_1_bn' % name)(l_1sub)
            add3 = Add()([l, l_1sub])
        else:
            add3 = l

        if h_i1sub is not None:
            avg_p1 = AveragePooling2D(pool_size=(3, 3), strides=(stride, stride), padding='same')(h_i1sub)
            # avg_p2 = AveragePooling2D(pool_size=(3, 3), strides=(stride, stride), padding='same')(h_i1sub)
            # add4 = Add()([avg_p1, avg_p2])
            add4 = avg_p1

            l_1sub_1 = SeparableConv2D(filter_o, (3, 3), padding='same', name='%s_sep3x3_sub_2' % name, trainable=trainable, depthwise_regularizer=OrthLocalRegSep2D)(h_i1sub)
            l_1sub_1 = Activation('relu', name='%s_sep3x3_sub_2_relu' % name)(l_1sub_1)
            l_1sub_1 = BatchNormalization(name='%s_sep3x3_sub_2_bn' % name)(l_1sub_1)
            l_1sub_2 = SeparableConv2D(filter_o, (5, 5), padding='same', name='%s_sep5x5_sub_2' % name, trainable=trainable, depthwise_regularizer=OrthLocalRegSep2D)(h_i1sub)
            l_1sub_2 = Activation('relu', name='%s_sep5x5_sub_2_relu' % name)(l_1sub_2)
            l_1sub_2 = BatchNormalization(name='%s_sep5x5_sub_2_bn' % name)(l_1sub_2)
            add5 = Add()([l_1sub_1, l_1sub_2])

            result = concatenate([add1, add2, add3, add4, add5])

            if is_tail:
                result = Conv2D(filter_o, (1, 1), strides=(1, 1), padding='same', name='%s_result1x1_align' % name, trainable=trainable, kernel_regularizer=OrthLocalReg2D)(result)
                result = Activation('relu', name='%s_result1x1_align_relu' % name)(result)
                result = BatchNormalization(name='%s_result1x1_align_bn' % name)(result)
            return result
        else:
            result = concatenate([add1, add2, add3])

            if is_tail:
                result = Conv2D(filter_o, (1, 1), strides=(1, 1), padding='same', name='%s_result1x1_align' % name, trainable=trainable, kernel_regularizer=OrthLocalReg2D)(result)
                result = Activation('relu', name='%s_result1x1_align_relu' % name)(result)
                result = BatchNormalization(name='%s_result1x1_align_bn' % name)(result)
            return result


    def ReductionCell(hi, h_i1sub, filter_i, filter_o, stride=2, name="NAS", is_tail=False):
        """
        adjust feature size & channel size
        """
        h_i1sub = _adjust_block(h_i1sub, hi, filter_o, 0, name)

        hi = Conv2D(filter_o, (1, 1), padding='same', name='%s_hi_align' % name, trainable=trainable, kernel_regularizer=OrthLocalReg2D)(hi)
        hi = Activation('relu', name='%s_hi_align_relu' % name)(hi)
        hi = BatchNormalization(name='%s_hi_align_bn' % name)(hi)

        """
        SubLayer 1
        """
        l = SeparableConv2D(filter_o, (5, 5), strides=(stride, stride), padding='same', name='%s_sep5x5_1' % name, trainable=trainable, depthwise_regularizer=OrthLocalRegSep2D)(hi)
        l = Activation('relu', name='%s_sep5x5_1_relu' % name)(l)
        l = BatchNormalization(name='%s_sep5x5_1_bn' % name)(l)

        if h_i1sub is not None:
            l_1sub = SeparableConv2D(filter_o, (7, 7), strides=(stride, stride), padding='same', name='%s_sep7x7_sub_1' % name, trainable=trainable, depthwise_regularizer=OrthLocalRegSep2D)(h_i1sub)
            l_1sub = Activation('relu', name='%s_sep7x7_sub_1_relu' % name)(l_1sub)
            l_1sub = BatchNormalization(name='%s_sep7x7_sub_1_bn' % name)(l_1sub)
            add1 = Add()([l, l_1sub])
        else:
            add1 = l

        l = SeparableConv2D(filter_o, (3, 3), strides=(stride, stride), padding='same', name='%s_sep3x3_1' % name, trainable=trainable, depthwise_regularizer=OrthLocalRegSep2D)(hi)
        l = Activation('relu', name='%s_sep3x3_1_relu' % name)(l)
        l = BatchNormalization(name='%s_sep3x3_1_bn' % name)(l)

        if h_i1sub is not None:
            l_1sub = SeparableConv2D(filter_o, (7, 7), strides=(stride, stride), padding='same', name='%s_sep7x7_sub_2' % name, trainable=trainable, depthwise_regularizer=OrthLocalRegSep2D)(h_i1sub)
            l_1sub = Activation('relu', name='%s_sep7x7_sub_2_relu' % name)(l_1sub)
            l_1sub = BatchNormalization(name='%s_sep7x7_sub_2_bn' % name)(l_1sub)
            add2 = Add()([l, l_1sub])
        else:
            add2 = l

        l = AveragePooling2D(pool_size=(3, 3), strides=(stride, stride), padding='same')(hi)
        if filter_i != filter_o:
            l = Conv2D(filter_o, (1, 1), strides=(1, 1), padding='same', name='%s_sep1x1_align' % name, trainable=trainable)(l)
            l = Activation('relu', name='%s_sep1x1_align_relu' % name)(l)
            l = BatchNormalization(name='%s_sep1x1_align_bn' % name)(l)
        if h_i1sub is not None:
            l_1sub = SeparableConv2D(filter_o, (5, 5), strides=(stride, stride), padding='same', name='%s_sep5x5_sub_1' % name, trainable=trainable, depthwise_regularizer=OrthLocalRegSep2D)(h_i1sub)
            l_1sub = Activation('relu', name='%s_sep5x5_sub_1_relu' % name)(l_1sub)
            l_1sub = BatchNormalization(name='%s_sep5x5_sub_1_bn' % name)(l_1sub)
            add3 = Add()([l, l_1sub])
        else:
            add3 = l

        """
        SubLayer 2
        """
        l = MaxPooling2D(pool_size=(3, 3), strides=(stride, stride), padding='same')(hi)
        if filter_i != filter_o:
            l = Conv2D(filter_o, (1, 1), strides=(1, 1), padding='same', name='%s_sep1x1_align_2' % name, trainable=trainable)(l)
            l = Activation('relu', name='%s_sep1x1_align_2_relu' % name)(l)
            l = BatchNormalization(name='%s_sep1x1_align_2_bn' % name)(l)
        l_1sub = SeparableConv2D(filter_o, (3, 3), padding='same', name='%s_sep3x3_2' % name, trainable=trainable, depthwise_regularizer=OrthLocalRegSep2D)(add1)
        l_1sub = Activation('relu', name='%s_sep3x3_2_relu' % name)(l_1sub)
        l_1sub = BatchNormalization(name='%s_sep3x3_2_bn' % name)(l_1sub)
        add4 = Add()([l, l_1sub])

        l = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(add1)
        add5 = Add()([l, add2])

        result = concatenate([add3, add4, add5])

        if is_tail:
            result = Conv2D(filter_o, (1, 1), strides=(1, 1), padding='same', name='%s_result1x1_align' % name, trainable=trainable, kernel_regularizer=OrthLocalReg2D)(result)
            result = Activation('relu', name='%s_result1x1_align_relu' % name)(result)
            result = BatchNormalization(name='%s_result1x1_align_bn' % name)(result)
        return result

    inputs = l = Input((img_size, img_size, 3), name='input')

    l = Conv2D(24, (3, 3), strides=(2, 2), padding='same', name='conv11', trainable=trainable, kernel_regularizer=OrthLocalReg2D)(inputs)
    l = Activation('relu', name='conv11_relu')(l)
    l = BatchNormalization(name='conv11_bn')(l)

    l2 = InvConv2D(8, (3, 3), strides=(2, 2), padding='same', name='inv_conv11', trainable=trainable, kernel_regularizer=OrthLocalReg2D)(inputs)
    l2 = Activation('relu', name='inv_conv11_relu')(l2)
    l2 = BatchNormalization(name='inv_conv11_bn')(l2)

    l = stem = concatenate([l, l2])

    # l = Conv2D(48, (3, 3), padding='same', name='conv12', trainable=trainable, kernel_regularizer=OrthLocalReg2D)(l)

    l = NAS_Redu_1 = ReductionCell(l, None, 32, 64, name="NAS_Redu_1")
    l = NAS_Redu_2 = ReductionCell(l, stem, 64, 128, name="NAS_Redu_2")
    l = NAS_Norm_3 = NormalCell(l, NAS_Redu_1, 128, 128, name="NAS_Norm_3")
    l = NAS_Norm_4 = NormalCell(l, NAS_Redu_2, 128, 128, name="NAS_Norm_4")

    l = NAS_Redu_5 = ReductionCell(l, NAS_Norm_3, 128, 192, name="NAS_Redu_5")
    l = NAS_Norm_6 = NormalCell(l, NAS_Norm_4, 192, 192, name="NAS_Norm_6", is_tail=True)
    l = CapsuleRouting(192, 192, 169, 169, name='cp0', reshape_cnn=True)(l)
    l = NAS_Norm_7 = NormalCell(l, NAS_Redu_5, 192, 192, name="NAS_Norm_7")

    l = NAS_Redu_8 = ReductionCell(l, NAS_Norm_6, 192, 384, name="NAS_Redu_8")
    l = NAS_Norm_9 = NormalCell(l, NAS_Norm_7, 384, 384, name="NAS_Norm_9", is_tail=True)
    l = CapsuleRouting(384, 384, 49, 49, name='cp1', reshape_cnn=True)(l)
    l = NAS_Norm_10 = NormalCell(l, NAS_Redu_8, 384, 384, name="NAS_Norm_10", use_deform=True)


    # l = ConvOffset2D(384, name='conv33_offset', kernel_regularizer=OrthLocalReg2D)(l, use_resam=True)
    l = NAS_Redu_11= ReductionCell(l, NAS_Norm_9, 384, 768, name="NAS_Redu_11")
    l = NAS_Norm_12 = NormalCell(l, NAS_Norm_10, 768, 768, name="NAS_Norm_12", is_tail=True)
    l = CapsuleRouting(768, 768, 16, 16, name='cp2', reshape_cnn=True,)(l)
    l = NAS_Norm_13 = NormalCell(l, NAS_Redu_11, 768, 768, name="NAS_Norm_13", use_deform=False, is_tail=True)
    # l = NAS_Redu_14= ReductionCell(l, NAS_Norm_13, 768, 1024, name="NAS_Redu_14")

    # tt = l = SpatialPyramidPooling([1, 2, 4], input_shape=[None, None, 768])(l)
    l = GlobalAveragePooling2D()(l)

    # l = CapsuleRouting(768, 768, 16, 8, name='cp1', reduce_max=False)(l)
    # l = CapsuleRouting(768, 256, 8, 8, name='cp2')(l)
    # l = CapsuleRouting(256, 128, 8, 8, name='cp3')(l)
    # l = CapsuleRouting(128, 32, 8, 4, name='cp4')(l)
    # l = CapsuleRouting(32, class_num, 4, 4, name='cp5', reduce_max=True)(l)
    # l = Dense(256, name='fc1', trainable=trainable, kernel_regularizer=OrthLocalReg1D)(l)
    # l = Activation('relu', name='fc1_relu')(l)
    #
    # l = Dense(128, name='fc2', trainable=trainable, kernel_regularizer=OrthLocalReg1D)(l)
    # l = Activation('relu', name='fc2_relu')(l)
    #
    # l = Dense(class_num, name='fc3', trainable=trainable)(l)
    # l = Activation('relu', name='fc3_relu')(l)


    # l = MaxPooling2D(name='max_pool_final')(l)
    # l = Flatten(name='flatten_maxpool')(l)

    l = Dense(768, name='fc1', trainable=trainable, kernel_regularizer=OrthLocalReg1D)(l)
    l = Activation('relu', name='fc1_relu')(l)
    
    l = Dense(256, name='fc2', trainable=trainable, kernel_regularizer=OrthLocalReg1D)(l)
    l = Activation('relu', name='fc2_relu')(l)
    
    l = Dense(class_num, name='fc3', trainable=trainable)(l)
    outputs = Activation('softmax', name='out')(l)

    return inputs, outputs
