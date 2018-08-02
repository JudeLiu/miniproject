from imageio import imwrite
import numpy as np
import pickle
import os
from skimage.transform import resize
from scipy.ndimage.interpolation import rotate
# from matplotlib.pyplot import imshow
from time import time

DATA_DIR = '../cifar-10-batches-py'
CROP_H, CROP_W = 24, 24
RANDOM_CROP_FACTOR = 10

class Dataset(object):
    def __init__(self, X, y, batch_size, train, mean=None, std=None, shuffle=False):
        assert X.shape[0] == y.shape[0], 'Got different numbers of data and labels'
        self._y = y
        self.batch_size, self.shuffle = batch_size, shuffle
        if train:
            self._mean = np.mean(X, axis=(0,1,2))
            self._std = np.std(X, axis=(0,1,2))
        else:
             self._mean = mean
             self._std = std 
        self._X = (X - self._mean) / self._std

    @property
    def X(self): return self._X
    
    @property
    def y(self): return self._y
    
    @property
    def mean(self): return self._mean

    @property
    def std(self): return self._std

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
    validation_percentage = kwargs.get('val_percent', 0.05)

    if augment and train:
        print('start augmentation')
        s = time()
        images, labels = image_augmentation(images, labels)
        print('done, cost ', time()-s)

    # mean_image = np.mean(images, axis=0).astype(np.float32)
    total_number = images.shape[0]
    num_validation = int(total_number * validation_percentage)
    num_training = total_number - num_validation
    print('Total: {}, #train: {}, #val: {}'.format(total_number, num_training, num_validation))
    if train:
        mask = range(num_training, num_training + num_validation)
        X_val = images[mask]
        y_val = labels[mask]
        mask = range(num_training)
        X_train = images[mask]
        y_train = labels[mask]
        train_dset = Dataset(X_train, y_train, train=True, batch_size=batch_size, shuffle=True)
        val_dset = Dataset(X_val, y_val, train=False, 
            mean=train_dset.mean,
            std=train_dset.std,
            batch_size=batch_size, shuffle=False)
        return train_dset, val_dset#, mean_image
    else:
        mean = kwargs.get('mean', None)
        std = kwargs.get('std', None)
        if augment:
            # images = image_augmentation(images)
            images, labels = augment_all_images(images, labels, resize, output_shape=(CROP_H, CROP_W))
        test_dset = Dataset(images, labels, train=False,
                            mean=mean, std=std,
                            batch_size=batch_size)
        return test_dset

def random_crop(image, **kwargs):
    shape = kwargs.get('shape', (CROP_H, CROP_W))
    h, w = shape
    x = np.random.randint(0, image.shape[1] - w)
    y = np.random.randint(0, image.shape[0] - h)
    return image[y:y+h, x:x+w, :]
    
def center_crop(image, **kwargs):
    shape = kwargs.get('shape', (CROP_H, CROP_W))
    c_h, c_w = shape
    h, w, _ = image.shape
    return image[(h - c_h)//2: (h + c_h)//2, (w - c_w)//2 : (w + c_w)//2, :]

def random_fliplr(image, **kwargs):
    prob = kwargs.get('prob', 0.5)
    if np.random.rand() < prob:
        image = np.fliplr(image)
    return image

def random_rotation(image, **kwargs):
    angle_range = kwargs.get('angle_range', (0, 180))
    h, w, _ = image.shape
    angle = np.random.randint(*angle_range)
    image = rotate(image, angle)
    image = resize(image, (h, w))
    return image

def augment_on_image(image, label, func, times=1, **kwargs):
    assert len(image.shape) == 3

    images = [func(image, **kwargs) for i in range(times)]
    images = np.stack(images, axis=0)

    labels = [label for i in range(times)]
    labels = np.stack(labels)

    return images, labels

def augment_all_images(images, labels, func, times=1, **kwargs):
    assert len(images.shape) == 4
    assert len(labels.shape) == 1
    ret_images, ret_labels = [], []

    for img, lab in zip(images, labels):
        i, l = augment_on_image(img, lab, func, times=times, **kwargs)
        ret_images.append(i)
        ret_labels.append(l)

    ret_images = np.concatenate(ret_images)
    ret_labels = np.concatenate(ret_labels)
    
    return ret_images, ret_labels

def image_augmentation(images, labels):
    """
    images: numpy array of shape [N, H, W, 3]
    labels: numpy arary of shape [N,]
    """
    # random crop
    images, labels = augment_all_images(images, labels, random_crop, times=5)

    # corner crop

    # horizontal flip
    images, labels = augment_all_images(images, labels, random_fliplr)

    # random rotation
    # images, labels = augment_all_images(images, labels, random_rotation)

    return images, labels


def test():
    images, labels = load_data(train=True)
    image_augmentation(images, labels)

if __name__ == '__main__':
    test()
