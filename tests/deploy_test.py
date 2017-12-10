import sys
sys.path.append('../scripts')

import argparse
import os
import numpy as np
import shutil
import tensorflow as tf
from termcolor import colored
from PIL import Image

from dataset_gen import NPZ_gen


parser = argparse.ArgumentParser()
parser.add_argument("-g", "--gpu", help="gpu string")
parser.add_argument("-w", "--weight", help=".pb model weight file.")
parser.add_argument("-i", "--img_dir", help="feed imgs inside folder.")

args = parser.parse_args()

if args.weight is None:
    print(colored('-w is necessary', color='red'))
    sys.exit(0)
elif '.pb' not in args.weight:
    print(colored('-w need to be .pb const graph file', color='red'))
    sys.exit(0)

# ---
# Config
GPU = args.gpu

img_size = 200
class_num = 6
batch_size = 32

n_train = 88880
# n_test = batch_size * 10
steps_per_epoch = int(np.ceil(n_train / batch_size))
validation_steps = 4000 // batch_size

dataset = NPZ_gen('./face_age_dataset', class_num, batch_size, 1000, dataset_size=n_train)
train_scaled_gen = dataset.get_some()
test_scaled_gen = dataset.get_val(num_batch=validation_steps)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
# config.gpu_options.per_process_gpu_memory_fraction = 0.3

with tf.Session(config=config) as sess:
    with tf.device(GPU):

        img_dir = [os.path.join(args.img_dir, d) for d in os.listdir(args.img_dir) if not os.path.isdir(os.path.join(args.img_dir, d))]
        inputs = []

        if os.path.exists('./test_result'):
            shutil.rmtree('./test_result')

        os.mkdir('./test_result')
        for i in range(class_num):
            os.mkdir('./test_result/%d' % i)

        # for img_p in img_dir:
        #     pil_img = Image.open(img_p)
        #     pil_img = pil_img.resize([img_size, img_size])
        #     np_img = np.asarray(pil_img, dtype=np.uint8)
        #     # np_img /= 255
        #     inputs.append(np_img)
        c = 0
        counter = {
            0: 0,
            1: 0,
            2: 0,
            3: 0,
            4: 0,
            5: 0,
        }
        ans_list = []

        for x, y in test_scaled_gen:
            for yy in y:
                counter[yy.argmax()] += 1
                ans_list.append(yy.argmax())
            if c > validation_steps:
                break
            inputs += [np_i for np_i in x]
            # print(inputs)
            # input()
            c += 1
        print(counter)

        c = 0
        pred_counter = {
            0: 0,
            1: 0,
            2: 0,
            3: 0,
            4: 0,
            5: 0,
        }
        pred_list = []

        with open(args.weight, 'rb') as f:
            print(colored("Loading PB const-graph:",color='green'))
            print("[%s]" % args.weight)

            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())

            names = [n.name for n in graph_def.node]
            x, y_pred = tf.import_graph_def(graph_def, return_elements=['input:0', 'y_pred:0'])

        for i in range(0, len(inputs), batch_size):
            if i + batch_size > len(inputs):
                break
            batch = np.array(inputs[i: i + batch_size], dtype=np.float32)
            pred = sess.run(y_pred, feed_dict={x: batch})

            for o, n in zip(pred, range(len(pred))):
                # print(o.shape)
                class_id = o.argmax()
                pred_list.append(class_id)
                pred_counter[class_id] += 1
                Image.fromarray(inputs[i + n].astype(np.uint8)).save('./test_result/%d/%d.jpg' % (class_id, i + n))

        print(pred_counter)
        ans_list = ans_list[:len(pred_list)]
        compare = list(map(lambda x: int(x[0] == x[1]), zip(ans_list, pred_list)))
        print(sum(compare) / len(compare))
