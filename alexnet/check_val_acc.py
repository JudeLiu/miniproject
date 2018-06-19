import tensorflow as tf
import argparse
import numpy as np
import os
from data_preprocess import load_data

os.environ["CUDA_VISIBLE_DEVICES"]="0"

def main():
    export_dir = 'train_log/'
    with tf.Session(graph=tf.Graph()) as sess:
        tf.saved_model.loader.load(sess, [tag_constants.TRAINING], export_dir)