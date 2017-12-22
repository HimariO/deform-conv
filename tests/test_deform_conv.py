import numpy as np
import keras.backend as K
import tensorflow as tf
from scipy.ndimage.interpolation import map_coordinates

from deform_conv.deform_conv import (
    tf_map_coordinates,
    sp_batch_map_coordinates, tf_batch_map_coordinates,
    sp_batch_map_offsets, tf_batch_map_offsets, add_batch_grid
)

from deform_conv.layers import ConvOffset2D


def test_tf_map_coordinates():
    np.random.seed(42)
    input = np.random.random((100, 100))
    coords = np.random.random((200, 2)) * 99

    sp_mapped_vals = map_coordinates(input, coords.T, order=1)
    tf_mapped_vals = tf_map_coordinates(
        K.variable(input), K.variable(coords)
    )
    assert np.allclose(sp_mapped_vals, K.eval(tf_mapped_vals), atol=1e-5)


def test_tf_batch_map_coordinates():
    np.random.seed(42)
    input = np.random.random((4, 100, 100))
    coords = np.random.random((4, 200, 2)) * 99

    sp_mapped_vals = sp_batch_map_coordinates(input, coords)
    tf_mapped_vals = tf_batch_map_coordinates(
        K.variable(input), K.variable(coords)
    )
    assert np.allclose(sp_mapped_vals, K.eval(tf_mapped_vals), atol=1e-5)


def test_tf_batch_map_offsets():
    np.random.seed(42)
    input = np.random.random((4, 100, 100))
    offsets = np.random.random((4, 100, 100, 2)) * 2

    sp_mapped_vals = sp_batch_map_offsets(input, offsets)
    tf_mapped_vals = tf_batch_map_offsets(
        K.variable(input), K.variable(offsets)
    )
    assert np.allclose(sp_mapped_vals, K.eval(tf_mapped_vals), atol=1e-5)


def test_tf_batch_map_offsets_grad():
    np.random.seed(42)
    input = np.random.random((4, 100, 100))
    offsets = np.random.random((4, 100, 100, 2)) * 2

    input = K.variable(input)
    offsets = K.variable(offsets)

    tf_mapped_vals = tf_batch_map_offsets(input, offsets)
    grad = K.gradients(tf_mapped_vals, input)[0]
    grad = K.eval(grad)
    assert not np.allclose(grad, 0)


def test_tf_resampler_layer():
    np.random.seed(42)
    x = np.random.random((4, 10, 10, 3))
    # offsets = np.random.random((4, 20, 20, 6)) * 2
    x = K.variable(x)

    layer = ConvOffset2D(3)

    ori_offset = K.eval(layer(x))
    tf_offset = K.eval(layer(x, use_resam=True))

    print(ori_offset.shape)
    print(tf_offset.shape)
    print('-' * 100)
    print(ori_offset[0, :, :, 0])
    print('+' * 100)
    print(tf_offset[0, :, :, 0])
    print('-' * 100)

    print('-' * 100)
    print(ori_offset[0, :, :, 1])
    print('+' * 100)
    print(tf_offset[0, :, :, 1])
    print('-' * 100)

    assert hasattr(tf.contrib, 'resampler')
    assert np.allclose(ori_offset, tf_offset, atol=1e-4)


def test_tf_resampler():
    np.random.seed(42)
    input = np.random.random((4, 100, 100))
    offsets = np.random.random((4, 100, 100, 2))

    sp_mapped_vals = sp_batch_map_offsets(input, offsets).reshape((4, 100, 100, 1))

    offsets = add_batch_grid(input, offsets)
    tf_mapped_vals = tf.contrib.resampler.resampler(
        K.variable(input.reshape((4, 100, 100, 1))), K.variable(offsets)
    )
    assert np.allclose(sp_mapped_vals, K.eval(tf_mapped_vals), atol=1e-5)


def test_croodinate():
    coords = np.random.random((10, 5, 5, 2))
    coords = K.variable(coords)

    x = np.random.random((10, 5, 5, 1))
    x = K.variable(x)

    coord1 = add_batch_grid(x, coords)  # (10, 5, 5, 2)
    # chnagethis 'tf_batch_map_offsets' to return coordiante or this test wont work!!!
    coord2 = tf_batch_map_offsets(x, coords, get_coord=True)  # (10, 25, 2)

    coord1 = K.eval(coord1).reshape([10, -1, 2])
    coord2 = K.eval(coord2)
    assert np.allclose(coord1, coord2, atol=1e-5)
