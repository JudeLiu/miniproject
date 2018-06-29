import numpy as np
import tensorflow as tf
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

TRAIN_DICT_PATH = 'feat.npy'
VAL_DICT_PATH = 'feat_val.npy'
TEST_DICT_PATH = 'feat_test.npy'


def load_dict(npy_path):
    d = np.load(npy_path).item()
    feat, pred, gt = d['feat'], d['pred'], d['gt']
    return feat, pred, gt

def train_classifier():
    train_dict = np.load(TRAIN_DICT_PATH).item()
    val_dict = np.load(VAL_DICT_PATH).item()
    test_dict = np.load(TEST_DICT_PATH).item()
    train_feat = np.array(train_dict['feat']).reshape((-1, 512)).astype(np.float32)
    val_feat = np.array(val_dict['feat']).reshape((-1, 512)).astype(np.float32)
    test_feat = np.array(test_dict['feat']).reshape(-1, 512).astype(np.float32)

    train_input_fn = tf.estimator.inputs.numpy_input_fn(
                    x={'x': train_feat}, 
                    y=np.array(train_dict['gt']),
                    batch_size=128, num_epochs=None,
                    shuffle=True,)
    feature_columns = tf.feature_column.numeric_column('x', shape=[512])


    # summary_hook = tf.train.SummarySaverHook(save_steps=100,
    #                                         output_dir='log/clf/',
    #                                         summary_op=)
    clf = tf.estimator.LinearClassifier([feature_columns], n_classes=10,)
    clf.train(input_fn=train_input_fn,
                # hooks=[tf.train.SummarySaverHook],
                steps=10000)

    val_fn = tf.estimator.inputs.numpy_input_fn(
                    x={'x': val_feat}, 
                    y=np.array(val_dict['gt']),
                    num_epochs=1,
                    shuffle=False)

    val_acc = clf.evaluate(input_fn=val_fn)['accuracy']
    print('val acc = ', val_acc)

if __name__ == '__main__':
    train_classifier()