# import tensorflow as tf
from imageio import imwrite
import numpy as np
import pickle

DATA_DIR = 'cifar-10-batches-py'
import os
os.listdir(DATA_DIR)
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def vec_to_mat3d(vec, image_shape=(32, 32, 3)):
    if isinstance(vec, np.ndarray):
        if len(vec.shape) == 1:
            channels = np.split(vec, 3, axis=0)
            channels = [np.resize(c, image_shape[:-1]) for c in channels]
            mat = np.stack(channels, axis=2)
            # mat[..., 0] = np.reshape(vec[:1024], image_shape[:-1])
            # mat[..., 1] = np.reshape(vec[1024:2048], image_shape[:-1])
            # mat[..., 2] = np.reshape(vec[2048:], image_shape[:-1])
            return mat
        elif len(vec.shape) == 2:
            channels = np.split(vec, 3, axis=1)
            channels = [np.resize(c, (vec.shape[0], *image_shape[:-1])) for c in channels]
            mat = np.stack(channels, axis=3)
            # N = vec.shape[0]
            # mat = np.zeros((N, 32, 32, 3))
            # for i, v in enumerate(vec):
            #     mat[i, ...] = vec_to_mat3d(v)
            return mat
    else:
        raise TypeError('type is {}'.format(type(vec)))

def test():
    raw_pics = unpickle('{}/test_batch'.format(DATA_DIR))
    arr = raw_pics[b'data'][100]
    # img = np.zeros(32,32,3)
    print(arr.shape)
    img = vec_to_mat3d(arr)
    imwrite('img.png', img)

def load_data(train=True, raw=False):
    if train:
        data_path = ['{}/data_batch_{}'.format(DATA_DIR, i) for i in range(1,6)]
    else:
        data_path = ['{}/test_batch'.format(DATA_DIR)]

    raw_data = []
    labels = []
    for p in data_path:
        raw_data_dict = unpickle(p)
        raw_data.append(raw_data_dict[b'data'])
        labels += raw_data_dict[b'labels']

    raw_data = [np.array(d) for d in raw_data]
    raw_data = np.concatenate(raw_data, axis=0)

    if raw:
        data = raw_data
    else:
        data = vec_to_mat3d(raw_data)
    labels = np.array(labels)
    return data, labels

def test_load_data():
    data, labels = load_data()
    print(data.shape)
    print(len(labels))
    meta = unpickle('%s/batches.meta'%DATA_DIR)[b'label_names']
    for i in range(len(data)):
        imwrite('test/img_{}_{}.png'.format(i, meta[labels[i]].decode('utf8')), data[i])

