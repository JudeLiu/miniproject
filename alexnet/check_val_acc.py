import tensorflow as tf
import argparse
import numpy as np
import os
from data_preprocess import load_data, prepare_dataset
from alexnet import check_accuracy

os.environ["CUDA_VISIBLE_DEVICES"]="0"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save-dir', '-s', type=str, default='model')
    parser.add_argument('--val', '-v', dest='val', action='store_true')
    parser.add_argument('--test', '-t', dest='val', action='store_false')
    parser.set_defaults(val=False)
    args = parser.parse_args()

    if args.val:
        images, labels = load_data(True)
        _, val_dset = prepare_dataset(images, labels, train=True, augment=False)
    else:
        images, labels = load_data(False)
        test_dset = prepare_dataset(images, labels, train=False)
    dataset = val_dset if args.val else test_dset

    graph = tf.Graph()
    with graph.as_default():
        with tf.Session(graph=graph) as sess:
            tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], args.save_dir)
            X = graph.get_tensor_by_name('images_ph:0')
            logits_op = graph.get_tensor_by_name('dense/BiasAdd:0')
            acc, num_correct, num_samples = check_accuracy(sess, dataset, X, logits_op)  
        print('{} acc: {:.2%} ({}/{})'.format('val' if args.val else 'test', acc, num_correct, num_samples))

if __name__ == '__main__':
    main()    