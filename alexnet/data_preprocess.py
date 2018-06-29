import tensorflow as tf
from imageio import imwrite
import numpy as np
import pickle
import os
from skimage import transform

DATA_DIR = 'cifar-10-batches-py'
CROP_H, CROP_W = 24, 24

class Dataset(object):
    def __init__(self, X, y, batch_size, shuffle=False):
        assert X.shape[0] == y.shape[0], 'Got different numbers of data and labels'
        self.X, self.y = X, y
        self.batch_size, self.shuffle = batch_size, shuffle

    def __iter__(self):
        N, B = self.X.shape[0], self.batch_size
        idxs = np.arange(N)
        if self.shuffle:
            np.random.shuffle(idxs)
        return iter((self.X[i:i+B], self.y[i:i+B]) for i in range(0, N, B))


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
    data.astype(np.float32)
    return data, labels

def prepare_dataset(images, labels, train, **kwargs):
    augment = kwargs.get('augment', False)
    batch_size = kwargs.get('batch_size', 64)
    num_training = kwargs.get('num_training', 49000)
    num_validation = kwargs.get('num_validation', 1000)

    if augment and train:
        images = image_augmentation(images)

    if train:
        mask = range(num_training, num_training + num_validation)
        X_val = images[mask]
        y_val = labels[mask]
        mask = range(num_training)
        X_train = images[mask]
        y_train = labels[mask]
        train_dset = Dataset(X_train, y_train, batch_size=batch_size, shuffle=True)
        val_dset = Dataset(X_val, y_val, batch_size=batch_size, shuffle=False)
        return train_dset, val_dset
    else:
        if augment:
            # images = image_augmentation(images)
            images = resize(images)
        test_dset = Dataset(images, labels, batch_size=batch_size)
        return test_dset

def test_load_data():
    data, labels = load_data()
    print(data.shape)
    print(len(labels))
    meta = unpickle('%s/batches.meta'%DATA_DIR)[b'label_names']
    for i in range(len(data)):
        imwrite('test/img_{}_{}.png'.format(i, meta[labels[i]].decode('utf8')), data[i])

def resize(images):
    images_resize = np.zeros((images.shape[0], CROP_H, CROP_W, 3))
    for i in range(images.shape[0]):
        images_resize[i, :, :, :] = transform.resize(images[i, :, :, :], (CROP_H, CROP_W))
    return images_resize

def random_crop(image):
    x = np.random.randint(0, image.shape[1] - CROP_W)
    y = np.random.randint(0, image.shape[0] - CROP_H)
    return image[y:y+CROP_H, x:x+CROP_W, :]
    
def image_augmentation(images):
    augmented = np.zeros((images.shape[0], CROP_H, CROP_W, 3))
    for i in range(images.shape[0]):
        augmented[i, :, :, :] = np.fliplr(random_crop(images[i, :, :, :]))

    return augmented