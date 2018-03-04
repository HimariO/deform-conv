import numpy as np
import os
import re
import random
import threading
import time
import colorsys
import sys

from queue import Queue
from PIL import Image
from skimage import transform
from skimage import util
from skimage import filters
from skimage import exposure
from termcolor import colored
from joblib import Parallel, delayed
import multiprocessing

class threadsafe_iter:
    """Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    """
    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            return self.it.__next__()


def threadsafe_generator(f):
    """A decorator that takes a generator function and makes it thread-safe.
    """
    def g(*a, **kw):
        return threadsafe_iter(f(*a, **kw))
    return g


class NPZ_gen:
    def __init__(self, dataset_dir, class_num, batch_size, epoch, dataset_size=None,
                 soft_onehot=True, hierarchy_onehot=False, gate=False, flip=True, flip_v=False, resize=None,
                 random_scale=None, random_resize=None, random_crop=None, random_noise=None, random_blur=False, random_gamma=None):

        # assert random_scale < .5 and random_scale > .0
        # assert random_resize < .5 and random_scale > .0
        # assert random_crop < .4 and random_crop > .0
        if random_noise is not None:
            assert random_noise < .05 and random_noise > .001
        if resize is not None:
            assert type(resize) is int

        self.data_preprocess = {
            'resize': resize,
            'soft_onehot': soft_onehot,
            'flip': flip,
            'flip_v': flip_v,
            'random_scale': random_scale,
            'random_resize': random_resize,
            'random_crop': random_crop,
            'random_noise': random_noise,
            'random_blur': random_blur,
            'random_gamma': random_gamma,
            'hierarchy_onehot': hierarchy_onehot,
            'gate': gate,
        }
        # self.datas = [os.path.join(dataset_dir, f) for f in os.listdir(dataset_dir) if '.npz' in f or '.npy' in f]
        self.datas, self.val_datas = self._scan_dir(dataset_dir)

        if len(self.datas) == 0:
            raise ValueError("Can't find any npz data inside folder: %s" % dataset_dir)
        elif len(self.val_datas) == 0 and False:
            raise ValueError("Can't find any npz Validate data inside folder: %s" % dataset_dir)
        else:
            # shuffle order of dataset except validation set(last npz).
            random.shuffle(self.datas)

        self.npz_num = len(self.datas)
        self.batch_size = batch_size
        self.epoch = epoch
        self.class_num = class_num

        self.output_img_num = 0
        self.dataset_size = 0
        self.next_one = None  # npz object or 'end' str

        self.reader_thread = threading.Thread(target=self._auto_load_npz, name="reader_thread[0]")
        self.reader_thread.daemon = True
        self.reader_queue = Queue()
        self.reader_waittime = 0
        self.reader_timeout = 30

        if dataset_size is None:
            for npz_path in self.datas:
                npz = self._load_npz(npz_path)
                self.dataset_size += len(npz.keys())
                print(npz_path, ': ', len(npz.keys()), ' img')
        else:
            self.dataset_size = dataset_size

        # random.shuffle(self.datas)
        print(colored("Get training datas: ", color='green'), self.datas)
        print(colored("Get validate datas: ", color='green'), self.val_datas)

        for npz_path in self.datas * (self.epoch * 2):
            self.reader_queue.put(npz_path)

        print("Init Generator[%s]" % dataset_dir)
        self.next_one = self._load_npz(self.datas[0])
        # print(self.next_one.keys())
        self.reader_thread.start()

    def __str__(self):
        return self.datas + " iteration[%d/%d]" % (self.output_img_num, self.dataset_size)

    def _scan_dir(self, path):
        train_npzs = [re.match('dataset_(\d+)(_\w+)*\.npz', f) for f in os.listdir(path=path)]
        val_npzs = [re.match('validate_(\d+)(_\w+)*\.npz', f) for f in os.listdir(path=path)]

        # [(f.string, int(f.group(1))) for f in train_npzs if f is not None]
        train_npzs = [os.path.join(path, f.string) for f in train_npzs if f is not None]
        val_npzs = [os.path.join(path, f.string) for f in val_npzs if f is not None]

        return train_npzs, val_npzs

    def _load_npz(self, path):
        nz = np.load(path)
        npz = {k: nz[k].tolist() for k in nz}
        return npz

    def _auto_load_npz(self):
        while True:
            if self.next_one is None:
                next_file = self.reader_queue.get()
                self.next_one = self._load_npz(next_file)
                self.reader_queue.task_done()
            elif self.next_one == 'end':
                break

            time.sleep(0.01)

    def _wait_load(self):
        single_wait = 0.1
        if self.reader_waittime < self.reader_timeout:
            if self.next_one is None:
                print('wait for next file.')
                self.reader_waittime += single_wait
                time.sleep(single_wait)
                return True  # you should wait
            else:
                self.reader_waittime = 0
                return False # you can go on.
        else:
            print('giveup waiting.')
            return False  # go on and trigger excepetion.

    @threadsafe_generator
    def get_some(self):
        # loop through different npy file
        num_cores = multiprocessing.cpu_count()
        with Parallel(n_jobs=num_cores) as parallel:
            for npz_path in self.datas * (self.epoch * 2):
                # npz = np.load(npz_path)
                while(self._wait_load()):
                    pass

                npz = self.next_one
                self.next_one = None
                keys = list(npz.keys())
                random.shuffle(keys)

                npz_size = len(npz.keys())
                self.output_img_num += npz_size

                # loop through datas inside npy file
                for i in range(0, npz_size, self.batch_size):
                    if (i + self.batch_size) <= npz_size:
                        B = self._process_data(npz, i, i + self.batch_size, keys, parallel, **self.data_preprocess)
                        X, Y = B[0], B[1]
                        yield X, Y
                    else:
                        break
                del npz

        self.next_one = 'end'

    @threadsafe_generator
    def get_val(self, num_batch=10):
        """
        num_batch: how many batch of data you want to use as validation data.
        """
        npz = self._load_npz(self.val_datas[0])
        val_size = len(npz.keys())
        keys = list(npz.keys())
        # random.shuffle(keys)

        if num_batch > val_size // self.batch_size:
            print("You can  have more than %d batch!" % (val_size // self.batch_size))

        pick_group = random.sample(range(0, val_size - self.batch_size, self.batch_size), num_batch)

        print('\n' + '-' * 100)
        print('Validate on %s\'s: %s' % (self.val_datas[0], str(pick_group)))
        print('-' * 100)

        num_cores = multiprocessing.cpu_count()
        with Parallel(n_jobs=num_cores) as parallel:
            for i in range(self.epoch * 2 * num_batch):
                pick_start = pick_group[i % num_batch]
                imgs, lab = self._process_data(
                    npz, pick_start, pick_start + self.batch_size, keys, parallel,
                    soft_onehot=self.data_preprocess['soft_onehot'],
                    hierarchy_onehot=self.data_preprocess['hierarchy_onehot'],
                    gate=self.data_preprocess['gate'],
                    resize=self.data_preprocess['resize'],
                )
                yield imgs, lab

    @staticmethod
    def _hierarchy_onehot(onehot, target_id, class_num):
        split = class_num // 2
        super_class = -1

        if target_id < split:
            super_class = 0
        else:
            super_class = 1

        super_class_onehot = np.zeros([2])
        super_class_onehot[super_class] = 1.
        return super_class_onehot

    @staticmethod
    def _run(item, args):

        keras_model = args['keras_model']
        flip = args['flip']
        flip_v = args['flip_v']
        resize = args['resize']

        random_scale = args['random_scale']
        random_crop = args['random_crop']
        random_resize = args['random_resize']
        random_noise = args['random_noise']
        random_blur = args['random_blur']
        random_gamma = args['random_gamma']

        crop_size = args['crop_size']
        crop_top = args['crop_top']
        crop_left = args['crop_left']

        class_num = args['class_num']
        soft_onehot = args['soft_onehot']
        hierarchy_onehot = args['hierarchy_onehot']
        gate = args['gate']
        new_size_scale = args['new_size_scale']

        tar_id = item['label']
        img_fp = item['img'] / 255. if not keras_model else item['img'].astype(np.float32)
        blur_level = int(10 * random.random()) if random_blur else 0

        if flip:
            img_fp = np.flip(img_fp, 1) if random.random() > 0.5 else img_fp

        if flip_v:
            img_fp = np.flip(img_fp, 0) if random.random() > 0.5 else img_fp
            # img_fp = np.flip(img_fp, 0) if random.random() > 0.5 else img_fp
        if random_scale:
            for channel in range(img_fp.shape[2]):
                img_fp[:, :, channel] *= 1 - (random_scale * random.random() - random_scale / 2)
            img_fp = img_fp.clip(min=0, max=255) if keras_model else img_fp.clip(min=0, max=1)

        if random_resize:
            new_size = int(new_size_scale * img_fp.shape[0])
            img_fp = transform.resize(img_fp, [new_size, new_size, 3], preserve_range=True)

        if random_crop:
            h, w, c = img_fp.shape
            crop_size_ = [int(h*f) for f in crop_size]

            img_fp = img_fp[crop_size_[0]:, :, :] if crop_top else img_fp[:h - crop_size_[0], :, :]
            img_fp = img_fp[:, crop_size_[1]:, :] if crop_left else img_fp[:, :w - crop_size_[1], :]

        if random_gamma:
            if random.random() > 0.5:
                gamma_delta = random_gamma * (random.random() - 0.5)
                img_fp = exposure.adjust_gamma(img_fp, 1 + gamma_delta)
                img_fp = img_fp.clip(min=0, max=255) if keras_model else img_fp.clip(min=0, max=1)

        if random_noise and random.random()  > 0.33:
            img_fp = img_fp / 255. if keras_model else img_fp
            img_fp = util.random_noise(img_fp, var=random_noise)
            img_fp = img_fp * 255. if keras_model else img_fp
            img_fp = img_fp.clip(min=0, max=255) if keras_model else img_fp.clip(min=0, max=1)

        if random_blur and random.random()  > 0.5:
            img_fp = img_fp / 255. if keras_model else img_fp
            img_fp = filters.gaussian(img_fp, sigma=blur_level, multichannel=True)
            img_fp = img_fp * 255. if keras_model else img_fp
            img_fp = img_fp.clip(min=0, max=255) if keras_model else img_fp.clip(min=0, max=1)

        if resize:
            img_fp = transform.resize(img_fp, [resize, resize, 3], preserve_range=True)

        img_onehot = np.zeros([class_num])

        # img_onehot[tar_id] = 1  # int
        """
        onehot
        """
        if tar_id >= 0 and tar_id < class_num:
            if soft_onehot:
                delta1 = (random.random() - 0.5) * 2 * 0.1 if False else 0
                delta2 = (random.random() - 0.5) * 2 * 0.1 if False else 0

                if tar_id != 0 and tar_id != class_num - 1:
                    img_onehot[tar_id] = 0.8 + delta1 + delta2  # int
                    img_onehot[tar_id - 1] = 0.1 + delta1
                    img_onehot[tar_id + 1] = 0.1 + delta2
                else:
                    img_onehot[tar_id] = 0.9 + delta1 # int
                    if tar_id == 0:
                        img_onehot[tar_id + 1] = 0.1 + delta1
                    else:
                        img_onehot[tar_id - 1] = 0.1 + delta1
            else:
                img_onehot[tar_id] = class_num
        else:  # unclassifiable sample
            if gate:
                img_onehot[:] = 1. / class_num
            else:
                img_onehot[-1] = 1.

        """
        Additional labels
        """
        if hierarchy_onehot:
            super_onehot = NPZ_gen._hierarchy_onehot(img_onehot, tar_id, class_num)
            img_onehot = [img_onehot, super_onehot]
        elif gate:
            if tar_id > 0 and tar_id < class_num:
                img_onehot = [img_onehot, np.ones([1])]
            else:
                blur_level_ = (10 - blur_level) / 10
                img_onehot = [img_onehot, np.array([blur_level_])]

        return img_fp, img_onehot

    def _process_data(self, npz, range_a, range_b, keys, parallel, resize=None, flip=False, flip_v=False, soft_onehot=True, hierarchy_onehot=False, gate=False,
                      keras_model=False, random_scale=None, random_resize=None, random_crop=None, random_noise=None, random_blur=False, random_gamma=False):
        # keys = list(npz.keys())
        input_vec = []
        output_vec = []

        crop_size = []
        crop_top = random.random() < 0.5
        crop_left = random.random() < 0.5
        new_size_scale = 1

        if random_resize:
            if random.random() < 0.333:
                new_size_scale = 1 - random_resize * random.random()
            else:
                new_size_scale = 1
            # elif random.random() > 0.333 and random.random() < 0.666:
            #     new_size_scale = 1 + random_resize * random.random() / 2
        if random_crop:
            crop_size = [random_crop * random.random()] * 2

        args = {
            'flip': flip,
            'flip_v': flip_v,
            'resize': resize,
            'soft_onehot': soft_onehot,
            'hierarchy_onehot': hierarchy_onehot,
            'gate': gate,
            'keras_model': keras_model,
            'random_scale': random_scale,
            'random_resize': random_resize,
            'random_noise': random_noise,
            'random_crop': random_crop,
            'random_blur': random_blur,
            'random_gamma': random_gamma,
            'crop_size': crop_size,
            'crop_top': crop_top,
            'crop_left': crop_left,
            'class_num': self.class_num,
            'new_size_scale': new_size_scale,
        }

        num_cores = multiprocessing.cpu_count()
        results = parallel(delayed(self._run)(npz[img_name], args) for img_name in keys[range_a: range_b])
        multi_output_vec = [[], []]

        for r in results:
            input_vec.append(r[0])
            if type(r[1]) is list:
                multi_output_vec[0].append(r[1][0])
                multi_output_vec[1].append(r[1][1])
                # output_vec.append(r[1])
            else:
                output_vec.append(r[1])

        if len(output_vec) == 0: # multi target
            return np.array(input_vec), [np.array(multi_output_vec[0]), np.array(multi_output_vec[1])]
        else:
            return np.array(input_vec), np.array(output_vec)

class NPZ_class_id(NPZ_gen):
     def _process_data(self, *args, **kwargs):
        X, Y = super(NPZ_class_id, self)._process_data(*args, **kwargs)
        dummy_target = np.zeros([self.batch_size, 256])
        label_id_vec = Y.argmax(axis=1)
        return [X, np.array(label_id_vec)], [Y, np.array(dummy_target)]

if __name__ == "__main__":
    i = 0
    # npz_2('../../val_set', resize=[200, 200])

    # mix_dataset('face_age_dataset')

    GEn = NPZ_gen('./face_age_dataset', 6, 32, 100, hierarchy_onehot=True, dataset_size=95000, random_noise=0.02)

    print('Out side')
    GEn.get_val()
    for I, O in GEn.get_some():
        print('Inside %d ' % i)
        print(I)
        print(I.shape)
        print(O.shape)
    #     # np.save('debug.npy', D[0])
        i += 1
        if i > 50:
            break
