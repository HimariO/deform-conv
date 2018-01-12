from __future__ import absolute_import, division


import tensorflow as tf
import keras.backend as K

from keras.layers import Conv2D, SeparableConv2D
from keras.engine.topology import Layer
from keras.initializers import RandomNormal
from deform_conv.deform_conv import tf_batch_map_offsets, add_batch_grid


class ConvOffset2D(Conv2D):
    """ConvOffset2D

    Convolutional layer responsible for learning the 2D offsets and output the
    deformed feature map using bilinear interpolation

    Note that this layer does not perform convolution on the deformed feature
    map. See get_deform_cnn in cnn.py for usage
    """

    def __init__(self, filters, init_normal_stddev=0.01, **kwargs):
        """Init

        Parameters
        ----------
        filters : int
            Number of channel of the input feature map
        init_normal_stddev : float
            Normal kernel initialization
        **kwargs:
            Pass to superclass. See Con2D layer in Keras
        """

        self.filters = filters
        super(ConvOffset2D, self).__init__(
            self.filters * 2, (3, 3), padding='same', use_bias=False,
            kernel_initializer=RandomNormal(0, init_normal_stddev),
            **kwargs
        )

    def call(self, x, use_resam=False):
        """
        Return the deformed featured map
        x: (b, h, w, f) output of other conv layer.
        """
        # print('[debug] ', x)
        # x_shape = x.get_shape()
        x_shape = tf.shape(x)
        offsets = super(ConvOffset2D, self).call(x)

        # offsets: (b*c, h, w, 2)

        # x: (b*c, h, w)

        x = self._to_bc_h_w(x, x_shape)
        offsets = self._to_bc_h_w_2(offsets, x_shape)

        if hasattr(tf.contrib, 'resampler') and use_resam:
            offsets = add_batch_grid(x, offsets)

            x = tf.einsum('ijk->ikj', x)
            x = tf.expand_dims(x, axis=-1)
            # return offsets

            # x_offset: (bc, h, w, 1)
            x_offset = tf.contrib.resampler.resampler(x, offsets)
            # x_offset: (bc, h, w)
            x_offset = tf.squeeze(x_offset)
        else:
            # x_offset: (bc, hw)
            x_offset = tf_batch_map_offsets(x, offsets)

        # x_offset: (b, h, w, c)
        x_offset = self._to_b_h_w_c(x_offset, x_shape)

        return x_offset

    def compute_output_shape(self, input_shape):
        """Output shape is the same as input shape

        Because this layer does only the deformation part
        """
        return input_shape

    @staticmethod
    def _to_bc_h_w_2(x, x_shape):
        """(b, h, w, 2c) -> (b*c, h, w, 2)"""
        x = tf.transpose(x, [0, 3, 1, 2])
        x = tf.reshape(x, (-1, x_shape[1], x_shape[2], 2))
        return x

    @staticmethod
    def _to_bc_h_w(x, x_shape):
        """(b, h, w, c) -> (b*c, h, w)"""
        x = tf.transpose(x, [0, 3, 1, 2])
        x = tf.reshape(x, (-1, x_shape[1], x_shape[2]))
        return x

    @staticmethod
    def _to_b_h_w_c(x, x_shape):
        """(b*c, h, w) -> (b, h, w, c)"""
        x = tf.reshape(
            x, (-1, x_shape[3], x_shape[1], x_shape[2])
        )
        x = tf.transpose(x, [0, 2, 3, 1])
        return x


class SepConvOffset2D(ConvOffset2D, SeparableConv2D):
    """ConvOffset2D

    Convolutional layer responsible for learning the 2D offsets and output the
    deformed feature map using bilinear interpolation

    Note that this layer does not perform convolution on the deformed feature
    map. See get_deform_cnn in cnn.py for usage
    """

    def __init__(self, filters, init_normal_stddev=0.01, **kwargs):
        """Init

        Parameters
        ----------
        filters : int
            Number of channel of the input feature map
        init_normal_stddev : float
            Normal kernel initialization
        **kwargs:
            Pass to superclass. See Con2D layer in Keras
        """

        self.filters = filters
        super(ConvOffset2D, self).__init__(
            self.filters * 2, (3, 3), padding='same', use_bias=False,
            kernel_initializer=RandomNormal(0, init_normal_stddev),
            **kwargs
        )


class InvConv2D(Conv2D):
    def call(self, x):
        # biases_in = 0
        I = tf.sign(x)
        inv_x = 1 - tf.abs(x)

        return super(InvConv2D, self).call(tf.multiply(I, inv_x))


class SpatialPyramidPooling(Layer):
    """Spatial pyramid pooling layer for 2D inputs.
    See Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition,
    K. He, X. Zhang, S. Ren, J. Sun
    # Arguments
        pool_list: list of int
            List of pooling regions to use. The length of the list is the number of pooling regions,
            each int in the list is the number of regions in that pool. For example [1,2,4] would be 3
            regions with 1, 2x2 and 4x4 max pools, so 21 outputs per feature map
    # Input shape
        4D tensor with shape:
        `(samples, channels, rows, cols)` if dim_ordering='th'
        or 4D tensor with shape:
        `(samples, rows, cols, channels)` if dim_ordering='tf'.
    # Output shape
        2D tensor with shape:
        `(samples, channels * sum([i * i for i in pool_list])`
    """

    def __init__(self, pool_list, **kwargs):

        self.dim_ordering = K.image_dim_ordering()
        assert self.dim_ordering in {'tf', 'th'}, 'dim_ordering must be in {tf, th}'

        self.pool_list = pool_list

        self.num_outputs_per_channel = sum([i * i for i in pool_list])

        super(SpatialPyramidPooling, self).__init__(**kwargs)

    def build(self, input_shape):
        if self.dim_ordering == 'th':
            self.nb_channels = input_shape[1]
        elif self.dim_ordering == 'tf':
            self.nb_channels = input_shape[3]

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.nb_channels * self.num_outputs_per_channel)

    def get_config(self):
        config = {'pool_list': self.pool_list}
        base_config = super(SpatialPyramidPooling, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def call(self, x, mask=None):

        input_shape = K.shape(x)

        if self.dim_ordering == 'th':
            num_rows = input_shape[2]
            num_cols = input_shape[3]
        elif self.dim_ordering == 'tf':
            num_rows = input_shape[1]
            num_cols = input_shape[2]

        row_length = [K.cast(num_rows, 'float32') / i for i in self.pool_list]
        col_length = [K.cast(num_cols, 'float32') / i for i in self.pool_list]

        outputs = []

        if self.dim_ordering == 'th':
            for pool_num, num_pool_regions in enumerate(self.pool_list):
                for jy in range(num_pool_regions):
                    for ix in range(num_pool_regions):
                        x1 = ix * col_length[pool_num]
                        x2 = ix * col_length[pool_num] + col_length[pool_num]
                        y1 = jy * row_length[pool_num]
                        y2 = jy * row_length[pool_num] + row_length[pool_num]

                        x1 = K.cast(K.round(x1), 'int32')
                        x2 = K.cast(K.round(x2), 'int32')
                        y1 = K.cast(K.round(y1), 'int32')
                        y2 = K.cast(K.round(y2), 'int32')
                        new_shape = [input_shape[0], input_shape[1],
                                     y2 - y1, x2 - x1]
                        x_crop = x[:, :, y1:y2, x1:x2]
                        xm = K.reshape(x_crop, new_shape)
                        pooled_val = K.max(xm, axis=(2, 3))
                        outputs.append(pooled_val)

        elif self.dim_ordering == 'tf':
            for pool_num, num_pool_regions in enumerate(self.pool_list):
                for jy in range(num_pool_regions):
                    for ix in range(num_pool_regions):
                        x1 = ix * col_length[pool_num]
                        x2 = ix * col_length[pool_num] + col_length[pool_num]
                        y1 = jy * row_length[pool_num]
                        y2 = jy * row_length[pool_num] + row_length[pool_num]

                        x1 = K.cast(K.round(x1), 'int32')
                        x2 = K.cast(K.round(x2), 'int32')
                        y1 = K.cast(K.round(y1), 'int32')
                        y2 = K.cast(K.round(y2), 'int32')

                        new_shape = [input_shape[0], y2 - y1,
                                     x2 - x1, input_shape[3]]

                        x_crop = x[:, y1:y2, x1:x2, :]
                        xm = K.reshape(x_crop, new_shape)
                        pooled_val = K.max(xm, axis=(1, 2))
                        outputs.append(pooled_val)

        if self.dim_ordering == 'th':
            outputs = K.concatenate(outputs)
        elif self.dim_ordering == 'tf':
            #outputs = K.concatenate(outputs,axis = 1)
            outputs = K.concatenate(outputs)
            #outputs = K.reshape(outputs,(len(self.pool_list),self.num_outputs_per_channel,input_shape[0],input_shape[1]))
            #outputs = K.permute_dimensions(outputs,(3,1,0,2))
            #outputs = K.reshape(outputs,(input_shape[0], self.num_outputs_per_channel * self.nb_channels))

        return outputs


class ImageNorm(Layer):

    def __init__(self, **kwargs):
        super(ImageNorm, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        super(ImageNorm, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        mean, var = tf.nn.moments(x, [1, 2, 3])

        num_element_per_img = tf.reduce_prod(tf.shape(x)[1:])
        num_element_per_img = tf.cast(num_element_per_img, tf.float32)

        min_dev = tf.rsqrt(num_element_per_img)
        adjusted_stddev = tf.clip_by_value(tf.sqrt(var), min_dev, 1e10)

        for i in range(3):
          mean = tf.expand_dims(mean, axis=1)

        for i in range(3):
          adjusted_stddev = tf.expand_dims(adjusted_stddev, axis=1)

        return (x - mean) / adjusted_stddev

    def compute_output_shape(self, input_shape):
        return input_shape


class gated_softmax(Layer):

    def __init__(self, class_num, **kwargs):
        self.class_num = class_num
        super(gated_softmax, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        super(gated_softmax, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, xs):
        shape = xs.get_shape().as_list()
        assert shape[1] == self.class_num + 1
        gate = xs[:, 0:1]
        softmax = xs[:, 1:]
        return gate * softmax

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1] - 1)


class BilinearConv(Layer):

    def __init__(self, **kwargs):
        super(BilinearConv, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        super(BilinearConv, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, Xs):
        A, B = Xs
        shape_a = A.get_shape().as_list()
        shape_b = B.get_shape().as_list()


        return gate * softmax

    def compute_output_shape(self, input_shape):
        conv1, con2 = input_shape

        return (input_shape[0], input_shape[1] - 1)


class CapsuleRouting(Layer):

    def __init__(self, input_capsule, output_capsule, input_size, output_size, reduce_max=False, reduce_avg=False, reshape_cnn=False, **kwargs):
        self.input_capsule = input_capsule
        self.output_capsule = output_capsule
        self.input_size = input_size
        self.output_size = output_size

        assert (reduce_avg and reduce_max and reshape_cnn) == False
        self.reduce_max = reduce_max
        self.reduce_avg = reduce_avg
        self.reshape_cnn = reshape_cnn

        super(CapsuleRouting, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.route = self._weight_variable(
            [self.output_capsule, self.input_capsule],
            input_n=self.input_capsule,
            output_n=self.output_capsule,
            regularizer=OrthLocalReg1D,
            name='W_route'
        )

        self.reduce_dim = self._weight_variable(
            [self.output_capsule, self.input_size, self.output_size],
            name='W_redim'
        )

        # self.FT_F = self._weight_variable(
        #     [self.output_capsule, self.input_size, self.input_size],
        #     name='W_FT_F'
        # )
        super(CapsuleRouting, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):

        x = self._reshape(x)
        output = self._capsule_net(x, self.input_capsule, self.output_capsule, self.input_size, self.output_size)

        if self.reduce_max:
            output = tf.reduce_max(output, axis=2)
        elif self.reduce_avg:
            output = tf.reduce_mean(output, axis=2)
        elif self.reshape_cnn:
            fmap_size = int(self.output_size**0.5)
            output = tf.transpose(output, perm=[0, 2, 1])
            output = tf.reshape(output, [-1, fmap_size, fmap_size, self.output_capsule])
        return output

    def _reshape(self, x):
        if len(x.get_shape().as_list()) == 4:
            # rerange dims if input if CNN feature map
            x = tf.transpose(x, perm=[0, 3, 1, 2])
        return tf.reshape(x, [-1, self.input_capsule, self.input_size])

    def _weight_variable(self, shape, input_n=0, output_n=0, baise=0, trainable=True, regularizer=None, name='W'):
        weights = self.add_weight(
            name=name,
            shape=shape,
            initializer='uniform',
            trainable=trainable,
            regularizer=regularizer,
        )
        return weights

    def _capsule_net(self, capsules, input_capsule, output_capsule, input_size, output_size):
        # _route =  tf.clip_by_value(self.route, 0.0, 1.0)
        _route = self.route
        translate_r = tf.einsum('ij,ajb->aib', _route, capsules)

        capsules_ = tf.reshape(translate_r, [-1, output_capsule, 1, input_size])

        xW = tf.reshape(
            tf.einsum('bjk,abcj->abck', self.reduce_dim, capsules_),
            [-1, output_capsule, output_size]
        ) # [-1,b,output_size]

        # capsules_t = tf.transpose(capsules_, perm=[0,1,3,2])
        # xT_FT_F_x_ = tf.einsum('abjk,abcj->abck', capsules_t, tf.einsum('bjk,abcj->abck', self.FT_F, capsules_))

        # xT_FT_F_x = tf.reshape(xT_FT_F_x_, [-1, output_capsule, 1])
        # translate = xW + xT_FT_F_x
        translate = xW

        return tf.nn.relu(translate)

    def compute_output_shape(self, input_shape):
        # print('compute_output_shape:input_shape', input_shape)
        if input_shape is None: # when using reduce_max
            return input_shape
        elif self.reduce_max or self.reduce_avg:
            return (input_shape[0], self.output_capsule)
        elif self.reshape_cnn:
            fmap_size = int(self.output_size**0.5)
            return (input_shape[0], fmap_size, fmap_size, self.output_capsule)
        else:
            return (input_shape[0], self.output_capsule, self.output_size)


def OrthLocalReg2D(W, L=10., ratio=1e-2):
    """
    Local orthognal reguliation for 2D CNN filter
    W: (height, width, input_channel, filter_num)
    """
    W_s = tf.shape(W)

    W_mtx = tf.reshape(W,[-1, W_s[3]])  # (H*W*I, F)
    W_f_inner = tf.matmul(tf.transpose(W_mtx), W_mtx) # inner product of (W[:,i] @ W[:,j] | for j in N, for i in N)

    W_f_len = tf.norm(W_mtx, axis=0)  # norm of filter weight; shape (F,)
    W_f_len = tf.reshape(W_f_len, [-1, 1])
    W_ij_len = tf.matmul(W_f_len, tf.transpose(W_f_len)) # outer product of filters norm; shape(F, F)

    W_ij_cos = tf.div(W_f_inner, W_ij_len)

    cost = 1 + tf.exp(L * (W_ij_cos - 1))
    cost = tf.log(cost)
    cost = cost - tf.diag(tf.diag_part(cost))
    cost = tf.reduce_sum(cost)

    return cost * ratio


def OrthLocalRegSep2D(W, L=10., ratio=5e-4):
    """
    Local orthognal reguliation for "2D Separable CNN filter"
    W: (height, width, input_channel, depth_multiplier)

    We group parameter by (H*W, I*M), so vectors been compare to each other are much
    smaller compare to OrthLocalReg2D
    (filter_size^2 v.s. filter_size^2 * input_channelm, typical case will be: 3^2 v.s 3^2*256).
    Hence cost value will be much larger to start with.
    (easier to get similar vector value when you have fewer parameter).
    And you probably shouldn't apply this regularizer to 1*1 filter serparable conv.
    """
    W_s = tf.shape(W)

    W_mtx = tf.reshape(W,[-1, W_s[2] * W_s[3]])  # (H*W, I*M)
    W_f_inner = tf.matmul(tf.transpose(W_mtx), W_mtx) # inner product of (W[:,i] @ W[:,j] | for j in N, for i in N)

    W_f_len = tf.norm(W_mtx, axis=0)  # norm of filter weight; shape (F,)
    W_f_len = tf.reshape(W_f_len, [-1, 1])
    W_ij_len = tf.matmul(W_f_len, tf.transpose(W_f_len)) # outer product of filters norm; shape(F, F)

    W_ij_cos = tf.div(W_f_inner, W_ij_len)

    cost = 1 + tf.exp(L * (W_ij_cos - 1))
    cost = tf.log(cost)
    cost = cost - tf.diag(tf.diag_part(cost))
    cost = tf.reduce_sum(cost)

    return cost * ratio


def OrthLocalReg1D(W, L=10., ratio=5e-3):
    """
    Local orthognal reguliation for 1D FC weight
    W: (input_size, output_size) weight matrix of 1 layer fc-layer.
    """
    W_s = tf.shape(W)

    W_mtx = W  # (I, O)
    W_f_inner = tf.matmul(tf.transpose(W_mtx), W_mtx) # inner product of (W[:,i] @ W[:,j] | for j in N, for i in N)

    W_f_len = tf.norm(W_mtx, axis=0)  # norm of filter weight; shape (F,)
    W_f_len = tf.reshape(W_f_len, [-1, 1])
    W_ij_len = tf.matmul(W_f_len, tf.transpose(W_f_len)) # outer product of filters norm; shape(F, F)

    W_ij_cos = tf.div(W_f_inner, W_ij_len)

    cost = 1 + tf.exp(L * (W_ij_cos - 1))
    cost = tf.log(cost)
    cost = cost - tf.diag(tf.diag_part(cost))
    cost = tf.reduce_sum(cost)

    return cost * ratio
