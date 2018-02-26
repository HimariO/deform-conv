from __future__ import division
# %env CUDA_VISIBLE_DEVICES=0

import os
import argparse
import sys
import numpy as np
import tensorflow as tf
import keras.backend as K
from lxml import etree
from termcolor import colored

from keras.models import Model
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import multi_gpu_model

from deform_conv.layers import ConvOffset2D
from deform_conv.callbacks import TensorBoard, SpreadSheet
from deform_conv.cnn import *
from deform_conv.mnist import get_gen
from deform_conv.utils import make_parallel, montecarlo_prediction
from dataset_gen_ import *

from PIL import ImageFile, Image, ImageDraw
ImageFile.LOAD_TRUNCATED_IMAGES = True

parser = argparse.ArgumentParser()
parser.add_argument("-g", "--gpu", help="gpu string")
parser.add_argument("-w", "--weight", help=".h5 model weight file.")
parser.add_argument("-i", "--img_dir", help="feed imgs inside folder.")
parser.add_argument("-s", "--sample", help="feed imgs inside folder.")

args = parser.parse_args()


# ---
# Config
GPU_NUM = 1
GPU = args.gpu

class_num = 2
batch_size = 32 * GPU_NUM

def get_overlay_map(background, frontground, offset=[0, 0]):
    fg_np = np.array(frontground)
    fh, fw, fc = fg_np.shape

    bg_np = np.array(background)
    bg_np = bg_np[offset[0]: offset[0] + fh, offset[1]: offset[1] + fw, :]
    delta = np.abs(bg_np - fg_np).sum(axis=2)
    mask = np.greater(delta, 50).astype(np.uint8)
    # print("masked [%d/%d]" % (mask.sum(), np.prod(fg_np.shape[:2])))
    return mask * 180

def parse_xml(xml_path):
    tree = etree.parse(xml_path)
    root = tree.getroot()

    xml_size = root.find('size')
    xml_width = float(xml_size.find('width').text)
    xml_height = float(xml_size.find('height').text)

    boxes = []
    for j, object in enumerate(root.findall('object')):
        boxConfig = []
        box = object.find('bndbox')
        xmin = float(box.find('xmin').text)
        ymin = float(box.find('ymin').text)
        xmax = float(box.find('xmax').text)
        ymax = float(box.find('ymax').text)
        boxConfig += [xmin, ymin, xmax, ymax]
        boxes.append(boxConfig)
    return boxes

def read_group(group_dir, batch_size=32):
    anno = os.path.join(group_dir, 'Annotations')
    jpeg = os.path.join(group_dir, 'JPEGImages')
    xmls = [x for x in os.listdir(anno) if '.xml' in x]
    jpgs = [x for x in os.listdir(jpeg) if '.jpg' in x]

    output_crops = []
    output_boxes = []

    for xml in xmls:
        img_p = os.path.join(jpeg, os.path.basename(xml).replace('.xml', '.jpg'))
        boxes = parse_xml(os.path.join(anno, xml))
        imgitself = Image.open(img_p)

        for box in boxes:
            w = box[1] - box[0]
            h = box[3] - box[2]

            if w > h:
                box[2] -= (w - h) // 2
                box[3] += (w - h) // 2
            else:
                box[0] -= (h - w) // 2
                box[1] += (h - w) // 2
            box[:2] = np.array(box[:2]).clip(min=0, max=imgitself.width)
            box[2:] = np.array(box[2:]).clip(min=0, max=imgitself.height)

            crop = imgitself.crop([box[0], box[1], box[2], box[3]])
            crop = crop.resize([200, 200])
            output_crops.append(np.array(crop))
            output_boxes.append(box)

        yield np.array(output_crops), output_boxes, imgitself
        output_crops = []
        output_boxes = []

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

with tf.Session(config=config) as sess:
    with tf.device(GPU):
        K.set_session(sess)

        # ---
        # Deformable CNN
        inputs, outputs = get_large_deform_cnn(class_num, trainable=True, dropout_sample=True)
        model = Model(inputs=inputs, outputs=outputs)
        model.summary()

        if GPU_NUM > 1:
            model = make_parallel(model, GPU_NUM)
            # model = multi_gpu_model(model, gpus=GPU_NUM)

        if args.weight is not None:
            print(colored("[load_weight] %s" % args.weight, color='green'))
            model.load_weights(args.weight)

        # sys.exit(0)
        # input("Press enter to start training...")
        optim = Adam(1e-4)
        loss = categorical_crossentropy

        model.compile(optim, [loss], metrics=['accuracy'])


        img_gen = read_group(args.img_dir)
        img_boxvar = []
        base_img = None

        batch_c = 0
        max_var = -1
        min_var = 999

        for crop, boxes, img in img_gen:
            prediction_probabilities, prediction_variances = montecarlo_prediction(model, crop, 48)

            boxes_  = []
            for box, var in zip(boxes, prediction_variances):
                box.append(var)
                boxes_.append(box)

            img_boxvar.append({
                'img': img,
                'boxes_var': boxes_
            })

            print('prediction_probabilities: ', prediction_probabilities)
            print('prediction_variances: ', prediction_variances)
            batch_c += 1
            max_var = max([max_var, prediction_variances.max()])
            min_var = min([min_var, prediction_variances.min()])

        if not os.path.exists(os.path.join(args.img_dir, 'draw')):
            os.mkdir(os.path.join(args.img_dir, 'draw'))

        crop_mask_center_color = []
        for pair in img_boxvar:
            img = pair['img']
            if base_img is None:
                base_img = img

            boxes = pair['boxes_var']
            draw = ImageDraw.Draw(img)

            for box in boxes:
                cache_info = {
                    'color': 0,
                    'center': (0, 0),
                    'crop': None,
                    'mask': None,
                    'box': None,
                }

                x_center = (box[0] + box[2]) // 2
                y_center = (box[1] + box[3]) // 2
                color = int(226 * (1 - (box[4] - min_var) / max_var - min_var))
                draw.ellipse([x_center, y_center, x_center + 20, y_center + 20], fill='hsl(' + str(color) +', 100%, 50%)')

                bcrop = img.crop([box[0], box[1], box[2], box[3]])
                mask = get_overlay_map(base_img, bcrop, (int(box[1]), int(box[0])))

                cache_info['color'] = color
                cache_info['center'] = (x_center, y_center)
                cache_info['crop'] = bcrop
                cache_info['mask'] = mask
                cache_info['box'] = box

                crop_mask_center_color.append(cache_info)
            img.save(os.path.join(args.img_dir, 'draw', os.path.basename(img.filename)))

        # base_img_ref = base_img.copy()
        for info in crop_mask_center_color[::2]:
            box = info['box']
            base_img.paste(
                info['crop'],
                [int(box[0]), int(box[1])],
                Image.fromarray(info['mask'])
            )

        base_img_draw = ImageDraw.Draw(base_img)
        for info in crop_mask_center_color:
            x_center = info['center'][0]
            y_center = info['center'][1]
            color = info['color']
            base_img_draw.ellipse([x_center, y_center, x_center + 10, y_center + 10], fill='hsl(' + str(color) +', 100%, 50%)')

        base_img.save(os.path.join(args.img_dir, 'draw', 'summary.jpg'))
