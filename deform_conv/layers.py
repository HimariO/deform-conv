from __future__ import absolute_import, division


import tensorflow as tf
from tensorflow.python.framework import function
import keras.backend as K

from keras.layers import Conv2D, SeparableConv2D
from keras.engine.topology import Layer
from keras.initializers import RandomNormal
from keras.utils import conv_utils
from keras import regularizers

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

    def call(self, x, use_resam=True):
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

    def __init__(self, filters, init_normal_stddev=0.01, **kwargs):

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


class sparse_conv2d(Conv2D):

    def call(self, x):

        @function.Defun(tf.float32, tf.float32)
        def k_grad(op, dy):
            output_size = tf.shape(dy)

            dy_flat = tf.reshape(dy, [-1, output_size[3]])
            dy_flat = tf.transpose(dy_flat, perm=[1, 0])

            # values, indices = tf.nn.top_k(
            #     tf.abs(dy_flat),
            #     k=tf.reduce_prod(output_size[:2]) // 2
            # )

            # mean, var = tf.nn.moments(tf.abs(dy_flat))
            # gate_value = mean - var * 0.8
            gate_value = tf.reduce_mean(tf.abs(dy_flat))

            # mid = tf.fill(tf.shape(dy_flat), gate_value)
            mask = tf.greater(tf.abs(dy_flat), gate_value)
            mask = tf.cast(mask, tf.float32)

            spare_dy = tf.multiply(dy_flat, mask)
            spare_dy = tf.transpose(spare_dy, perm=[1, 0])
            spare_dy = tf.reshape(spare_dy, output_size)
            spare_dy += 0

            return spare_dy - spare_dy

        @function.Defun(tf.float32, grad_func=k_grad)
        def grad_filter(filter_w):
            return filter_w

        self.kernel = grad_filter(self.kernel)

        return super(sparse_conv2d, self).call(x)


class GaussianAtten(Layer):

    def __init__(self, **kwargs):
        super(GaussianAtten, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        super(GaussianAtten, self).build(input_shape)  # Be sure to call this somewhere!

    def gaussian_mask(u, s, r):
        """
        :param u: tf.Tensor, centre of the first Gaussian.
        :param s: tf.Tensor, standard deviation of Gaussians.
        :param d: tf.Tensor, shift between Gaussian centres.
        :param R: int, number of rows in the mask, there is one Gaussian per row.
        :param C: int, number of columns in the mask.
        """
        # indices to create centres
        R = tf.to_float(tf.reshape(tf.range(r), (1, tf.to_int32(r))))
        R = tf.matmul(tf.ones([tf.shape(u)[0], 1]), R)

        mask = tf.exp(-.5 * tf.square((R - u) / s))
        # we add eps for numerical stability
        # normalised_mask = mask / tf.reduce_sum(mask, 1, keep_dims=True) + 1e-8
        return mask

    def call(self, x):

        return x

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], input_shape[2], 1)


class ReduceMax2D(Layer):

    def __init__(self, **kwargs):
        super(ReduceMax2D, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        super(ReduceMax2D, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, A_B):
        A = tf.expand_dims(A_B[0], -1)
        B = tf.expand_dims(A_B[1], -1)
        AB = tf.concat([A, B], -1)

        return tf.reduce_max(AB, axis=-1)

    def compute_output_shape(self, input_shape):
        shape_A, shape_B = input_shape

        return shape_A


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



class CapsulePooling2D(Layer):
    def __init__(self, pool_size, num_steps, **kwargs):
        self.pool_size = pool_size
        self.strides = pool_size
        self.num_steps = num_steps
        super(CapsulePooling2D, self).__init__(**kwargs)

    def get_config(self):
        config = {
            'pool_size': self.pool_size,
            'num_steps': self.num_steps
        }
        base_config = super(CapsulePooling2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def build(self, input_shape):
        super(CapsulePooling2D, self).build(input_shape)

    @staticmethod
    def squash_fm(fm):
        v_lenght = tf.norm(fm, axis=-1, keep_dims=True)
        v_lenght = v_lenght + K.epsilon()
        fm = fm / (1 + v_lenght)
        return fm

    @staticmethod
    def normalize_pool_map(score_map, pool_size=(2, 2), strides=(2, 2), temperature=1):
        # compute safe softmax by pool operation
        padding = 'same'
        input_shape = K.int_shape(score_map)

        pooled_max_map = K.pool2d(
            score_map, pool_size=pool_size, strides=strides, pool_mode='max',
            padding=padding
        )
        max_map = K.resize_images(
            pooled_max_map,
            pool_size[0],
            pool_size[1],
            data_format='channels_last'
        )

        max_map = max_map[:, :input_shape[1], :input_shape[2], :]
        max_map = K.stop_gradient(max_map)
        score_map = score_map - max_map
        score_map = tf.exp(score_map / temperature)

        pooled_score_map = K.pool2d(
            score_map, pool_size=pool_size, strides=strides, pool_mode='avg',
            padding=padding
        )

        avg_score_map = K.resize_images(
            pooled_score_map,
            pool_size[0],
            pool_size[1],
            data_format='channels_last'
        )
        # fit caps columns to feature map
        avg_score_map = avg_score_map[:, :input_shape[1], :input_shape[2], :]

        return score_map / (avg_score_map + K.epsilon())

    @staticmethod
    def routing_step(feature_map, pool_size, strides, score_map):

        # Compute average vector
        if score_map is None:
            # in first iteration we cannot compute weights average
            # just take input feature map
            weighted_fm = feature_map
        else:
            # normalize scores
            normalized_scores = CapsulePooling2D.normalize_pool_map(
                score_map, pool_size=pool_size, strides=strides, temperature=2
            )
            weighted_fm = normalized_scores * feature_map


        avg_feature_map = K.pool2d(
            weighted_fm,
            pool_size=pool_size,
            strides=strides,
            pool_mode='avg', padding='same'
        )

        avg_feature_map = CapsulePooling2D.squash_fm(avg_feature_map)
        avg_feature_columns = K.resize_images(
            avg_feature_map,
            pool_size[0],
            pool_size[1],
            data_format='channels_last'
        )
        # fit caps columns to feature map
        input_shape = K.int_shape(feature_map)
        avg_feature_columns = avg_feature_columns[
                              :, :input_shape[1], :input_shape[2], :input_shape[3]]

        # compute V * Vs
        new_score_map = K.sum(feature_map * avg_feature_columns, -1, keepdims=True)

        if score_map is None:
            score_map = new_score_map
        else:
            score_map = score_map + new_score_map

        return score_map, weighted_fm

    @staticmethod
    def routing_pool(feature_map, pool_size, strides, num_steps):

        score_map = None
        steps_history = []
        weighted_fm = None
        for i in range(num_steps):
            score_map, weighted_fm = CapsulePooling2D.routing_step(
                feature_map,
                pool_size=pool_size,
                strides=strides,
                score_map=score_map
            )
            steps_history.append(score_map)

        output_feature_map = K.pool2d(
            weighted_fm,
            pool_size=pool_size,
            strides=strides,
            pool_mode='avg', padding='same'
        )
        return output_feature_map, steps_history

    def call(self, x):
        pool_size = self.pool_size
        strides = self.strides
        num_steps = self.num_steps
        x, _ = self.routing_pool(x, pool_size, strides, num_steps)
        return x

    def compute_output_shape(self, input_shape):
        padding = 'same'
        rows = input_shape[1]
        cols = input_shape[2]
        rows = conv_utils.conv_output_length(rows, self.pool_size[0],
                                             padding, self.strides[0])
        cols = conv_utils.conv_output_length(cols, self.pool_size[1],
                                             padding, self.strides[1])
        return (input_shape[0], rows, cols, input_shape[3])


class ScaleConv(Layer):

    def __init__(self, filter_num, filter_size, input_channel, branch=1, stride=[1, 1, 1, 1],
                 kernel_initializer='uniform', kernel_regularizer=None, padding='SAME', **kwargs):
        self.min_filter_size = 1

        assert filter_size - self.min_filter_size > 3
        assert branch % 2 == 1 and branch - 1 <= (filter_size - 1) // 2

        self.filter_num = filter_num
        self.filter_size = filter_size

        self.stride = stride
        self.input_channel = input_channel
        self.branch = branch
        self.kernel_regularizer = kernel_regularizer
        self.kernel_initializer = kernel_initializer

        super(ScaleConv, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(
            name='%s_kernel' % self.name,
            shape=[self.filter_size, self.filter_size, self.input_channel, self.filter_num],
            initializer=self.kernel_initializer,
            trainable=True,
            regularizer=self.kernel_regularizer,
        )

        self.resize_kernel = []
        mid_branch = self.branch // 2 + 1
        kernel_FHWC = tf.transpose(self.kernel, perm=[3, 0, 1, 2])

        for b in range(1, self.branch + 1):
            if b > mid_branch:
                new_size = [self.filter_size + 2 * abs(b - mid_branch)] * 2
            elif b < mid_branch:
                new_size = [self.filter_size - 2 * abs(b - mid_branch)] * 2
            else:
                new_size = [self.filter_size] * 2

            resize_kernel = tf.image.resize_images(kernel_FHWC, new_size)
            resize_kernel = tf.transpose(resize_kernel, perm=[1, 2, 3, 0])
            self.resize_kernel.append(resize_kernel)

        # self.branch_fc_kernel = []
        #
        if self.branch > 1:
            self.kernel_x1 = self.add_weight(
                name='%s_kernel_x1' % self.name,
                shape=[1, 1, self.filter_num * self.branch, self.filter_num],
                initializer='uniform',
                trainable=True,
            )

        super(ScaleConv, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        GAP = tf.reduce_mean(x, [1, 2])

        self.branch_fc_output = []
        self.branch_conv_output = []

        kernel_FHWC = tf.transpose(self.kernel, perm=[3, 0, 1, 2])
        mid_branch = self.branch // 2 + 1

        print('-' * 100)
        for resize_kernel in self.resize_kernel:
            # self.branch_fc_output.append(
            #     tf.layers.dense(GAP, 2, activation=tf.nn.sigmoid)
            # )
            #
            # new_size = self.branch_fc_output[-1] * tf.cast(tf.shape(self.kernel)[0:2] - self.min_filter_size, tf.float32)
            # new_size += self.min_filter_size
            # new_size = tf.cast(new_size, tf.int32)
            # if b > mid_branch:
            #     new_size = [self.filter_size + 2 * abs(b - mid_branch)] * 2
            # elif b < mid_branch:
            #     new_size = [self.filter_size - 2 * abs(b - mid_branch)] * 2
            # else:
            #     new_size = [self.filter_size] * 2
            #
            # resize_kernel = tf.image.resize_images(kernel_FHWC, new_size)
            # resize_kernel = tf.transpose(resize_kernel, perm=[1, 2, 3, 0])
            # print(new_size)

            self.branch_conv_output.append(
                tf.nn.conv2d(
                    x,
                    resize_kernel,
                    self.stride,
                    'SAME',
                    use_cudnn_on_gpu=True,
                    data_format='NHWC',
                    dilations=[1, 1, 1, 1],
                )
            )

        if self.branch == 1:
            return self.branch_conv_output[0]
        else:
            all_branch_out = tf.concat(self.branch_conv_output, 3)

            final_out = tf.nn.conv2d(
                all_branch_out,
                self.kernel_x1,
                [1, 1, 1, 1],
                'SAME',
                use_cudnn_on_gpu=True,
                data_format='NHWC',
                dilations=[1, 1, 1, 1],
            )
            return final_out

    def compute_output_shape(self, input_shape):
        # print('compute_output_shape:input_shape', input_shape)
        if input_shape[1]:
            new_h = len(range(0, input_shape[1], self.stride[1]))
        else:
            new_h = None

        if input_shape[2]:
            new_w = len(range(0, input_shape[2], self.stride[2]))
        else:
            new_w = None

        return (input_shape[0], new_h, new_w, self.filter_num)


def OrthLocalReg2D(W, L=10., ratio=1e-2, L2=None):
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

    if L2:
        l2_cost = regularizers.l2(L2)(W)
        return cost * ratio + l2_cost
    else:
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
