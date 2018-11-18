import os
import re
import random
import threading
import time
import colorsys
import sys
import glob
import multiprocessing as mp

import fire
import numpy as np
import pandas as pd
from queue import Queue
from PIL import Image
from skimage.transform import resize
from termcolor import colored


def create_classifiy_npz_dataset(root_path, resize=None):
    root_path = os.path.abspath(root_path)
    img_all_class = []

    for cd in os.listdir(root_path):
        class_dir_path = os.path.abspath(os.path.join(root_path, cd))
        if not os.path.isdir(class_dir_path):
            continue
        img_names = [{'file_name': f, 'class': cd} for f in os.listdir(class_dir_path)
                     if '.jpg' in f.lower() or '.png' in f.lower()]
        img_all_class += img_names

    random.shuffle(img_all_class)

    dataset = {}
    data_size = len(img_all_class)
    num_per_file = data_size // 10 if data_size // 10 < 10000 else 10000
    num_per_file = 10000
    output_counter = 0

    for img_n, i in zip(img_all_class, range(data_size)):
        abs_path = os.path.join(root_path, img_n['class'], img_n['file_name'])

        try:
            sys.stdout.write('\r [%d/%d] %s' % (i, data_size, abs_path))
            sys.stdout.flush()
            pil_img = Image.open(abs_path)

            if resize is not None:
                pil_img = pil_img.resize(resize, Image.ANTIALIAS)
            # pil_img = HSVColor(pil_img)
            img = np.array(pil_img, dtype=np.uint8)

            if len(img.shape) != 3:
                continue
        except KeyboardInterrupt:
            sys.exit(1)
        except:
            print("\n Encounter some issue when reading {%s}." % abs_path)
            continue

        dataset[img_n['file_name']] = {}
        dataset[img_n['file_name']]['img'] = img
        dataset[img_n['file_name']]['label'] = int(img_n['class'])

        if (i % num_per_file == 0 and i != 0) or i == data_size - 1:
            file_name = os.path.join(root_path, 'dataset_%d' % output_counter)
            print('\n Save to: ', file_name)
            np.savez(file_name, **dataset)
            output_counter += 1
            del dataset
            dataset = {}


def create_human_protein_npz_dataset(image_path='', anno_path='', output_path='',
                                     resize=250, num_split=10):
    all_images = glob.glob(os.path.join(image_path, '*.png'))
    classes_by_id = {}
    csv = pd.read_csv(anno_path)

    for img_id, cls in zip(csv['Id'], csv['Target']):
        classes_by_id[img_id] = [int(c) for c in cls.split(' ')]

    img_ids = list(classes_by_id.keys())
    img_id_splits = [img_ids[i::num_split] for i in range(num_split)]

    def run_split(id, img_id_splits):
        npz_dict = {}
        for count, k in enumerate(img_id_splits):
            color_layers = []
            for c in ['green', 'blue', 'yellow', 'red']:
                img_filename = os.path.join(image_path, k + '_' + c + '.png')
                np_img = np.asarray(
                    Image.open(img_filename).resize([resize, resize])
                )
                color_layers.append(np_img[:, :, np.newaxis])
            npz_dict[k] = {}
            npz_dict[k]['img'] = np.concatenate(color_layers, axis=-1)
            npz_dict[k]['label'] = classes_by_id[k]
            print('[%d] %d/%d' % (id, count, len(img_id_splits)))

        file_name = os.path.join(output_path, 'dataset_%d' % id)
        print('\n Save to: ', file_name)
        np.savez(file_name, **npz_dict)

    processes = [mp.Process(target=run_split, daemon=True, args=[i, img_id_splits[i]])
                 for i in range(len(img_id_splits))]
    for p in processes: p.start()
    for p in processes: p.join()


def mix_dataset(dir_path):

    def _load_npz(path):
        print('Load %s' % path)
        nz = np.load(path)
        npz = {k: nz[k].tolist() for k in nz}
        return npz

    train_npzs = [re.match('dataset_(\d+)(_\w+)*\.npz', f) for f in os.listdir(path=dir_path)]
    val_npzs = [re.match('validate_(\d+)(_\w+)*\.npz', f) for f in os.listdir(path=dir_path)]

    # train_npzs = [os.path.join(dir_path, f.string) for f in train_npzs if f is not None]
    # val_npzs = [os.path.join(dir_path, f.string) for f in val_npzs if f is not None]

    train_dataset = {
        "None": []
    }

    for n in train_npzs:
        if n is None:
            continue

        if n.group(2) is None:
            train_dataset['None'].append(os.path.join(dir_path, n.string))
        else:
            try:
                train_dataset[n.group(2)].append(os.path.join(dir_path, n.string))
            except KeyError:
                train_dataset[n.group(2)] = [os.path.join(dir_path, n.string)]

    print(train_dataset)

    set_size = [len(train_dataset[k]) for k in train_dataset.keys()]
    set_name = list(train_dataset.keys())
    npz_counter = 0

    for i in range(max(set_size)):
        mix_set = {}
        for j in range(len(set_size)):
            try:
                file_name = train_dataset[set_name[j]][i]
                dataset = _load_npz(file_name)
                mix_set.update(dataset)

            except IndexError:
                # this dataset have less npz than other.
                print('Skip 1 npz in \'%s\' dataset' % set_name[j])
                pass

        keys = list(mix_set.keys())
        random.shuffle(keys)

        new_npz = {}
        data_num = len(keys)
        for k, ii in zip(keys, range(data_num)):
            new_npz[k] = mix_set[k]
            sys.stdout.write("\r INSERT %d / %d " % (ii, data_num))
            sys.stdout.flush()

            if (ii != 0 and ii % 10000 == 0) or ii == data_num - 1:
                npz_counter += 1
                print('Saving dataset_%d_mix' % npz_counter)
                np.savez('dataset_%d_mix' % npz_counter, **new_npz)
                del new_npz
                new_npz = {}

if __name__ == '__main__':
    fire.Fire({
        'create_protein_dataset': create_human_protein_npz_dataset
    })
