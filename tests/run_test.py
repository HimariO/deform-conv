import test_deform_conv as test_deconv
import tensorflow as tf
from termcolor import colored

with tf.Session() as sess:
    with sess.as_default():
        test_deconv.test_tf_batch_map_coordinates()
        test_deconv.test_tf_resampler_layer()
        # test_deconv.test_croodinate()
        print(colored('PASS!', color='green'))
