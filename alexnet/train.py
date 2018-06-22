import tensorflow as tf
import argparse
import numpy as np
import os
import shutil
from data_preprocess import load_data, prepare_dataset
from alexnet import *

os.environ["CUDA_VISIBLE_DEVICES"]="0"

def check_accuracy(sess, dset, X, logits):
    num_correct, num_samples = 0, 0
    for x_batch, y_batch in dset:
        feed_dict = {X: x_batch}
        logits_np = sess.run(logits, feed_dict=feed_dict)
        y_pred = logits_np.argmax(axis=1)
        num_samples += x_batch.shape[0]
        num_correct += (y_pred == y_batch).sum()
    acc = float(num_correct) / num_samples
    
    return acc, num_correct, num_samples


def build_model(X, Y, lr, **kwargs):
    lrn = kwargs.get('lrn', False)
    full_model = kwargs.get('full_model', True)
    # build model, loss and optimizer
    model = AlexNet if full_model else AlexNetTruncated
    logits_op = model(lrn)(X)
    loss_op = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=Y, logits=logits_op)
    loss_op = tf.reduce_mean(loss_op, name='loss')
    correct_pred = tf.equal(tf.cast(tf.argmax(logits_op, axis=1), tf.int32), Y)
    acc_op = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='train_acc')

    optim = tf.train.AdamOptimizer(lr)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = optim.minimize(loss_op)

    return logits_op, loss_op, acc_op, train_op


def create_dir(path):
    if os.path.exists(path):
        try:
            shutil.rmtree(path)
            print('Remove existed dir: {}'.format(path))
        except OSError as e:
            print('Error: {} - {}.'.format(e.filename, e.strerror))

def train(args):
    images, labels = load_data(True)
    train_dset, val_dset = prepare_dataset(images, labels, True, augment=args.augment)
    # images, labels = load_data(False)
    # test_dset = prepare_dataset(images, labels, False)

    # learning_rate = (9+2*np.random.random(4))*1e-5 ,
    learning_rate = [9.025036551142157e-05]

    for lr in learning_rate:
        tf.reset_default_graph()
        graph = tf.Graph()
        
        with graph.as_default():
            with tf.device('/device:GPU:0'):
                if not args.resume:
                    # input placeholder
                    input_shape = (32, 32) if not args.augment else (24, 24)
                    X = tf.placeholder(tf.float32, [None, *input_shape, 3], name='images_ph')
                    Y = tf.placeholder(tf.int32, [None], name='labels_ph')
                    logits_op, loss_op, acc_op, train_op = build_model(X, Y, lr, lrn=args.lrn, full_model=args.full_model)                                 

            if not args.resume:
                # add variables to summary
                log_dir = '{}/lr-{:.8f}'.format(args.log_dir, lr)
                create_dir(log_dir)

                tf.summary.scalar('loss', loss_op)
                tf.summary.scalar('train_acc', acc_op)
                tf.summary.histogram('loss', loss_op)
                summary_op = tf.summary.merge_all()
                summary_writer = tf.summary.FileWriter(log_dir, graph)

                # save builder
                save_dir = '{}/lr-{:.8f}'.format(args.save_dir, lr) if not args.resume else args.save_dir
                create_dir(save_dir)

            # run training session
            config = tf.ConfigProto(log_device_placement=False)
            with tf.Session(graph=graph, config=config) as sess:
                if not args.resume:
                    sess.run(tf.global_variables_initializer())
                else:
                    print('Resume from {}'.format(save_dir))
                    tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], save_dir)
                    X = tf.get_tensor_by_name('images_ph:0')
                    Y = tf.get_tensor_by_name('labels_ph:0')
                    loss_op = tf.get_tensor_by_name('')
                print('lr={}'.format(lr)) 
                cnt = 0

                for e in range(args.epochs):
                    print('Epoch {}'.format(e))
                    for i, (x_batch, y_batch) in enumerate(train_dset):
                        loss, _, acc, summary = sess.run([loss_op, train_op, acc_op, summary_op], feed_dict={X: x_batch, Y: y_batch})
                        summary_writer.add_summary(summary, cnt)
                        cnt += 1
                        if i % args.print_every == 0:
                            print('loss:{:.4f}, acc:{:.2%}'.format(loss, acc))

                    acc, num_correct, num_samples = check_accuracy(sess, val_dset, X, logits_op)
                    print('val acc: {:.2%} ({}/{})'.format(acc, num_correct, num_samples))
            
                tf.saved_model.simple_save(sess, save_dir, 
                                    inputs={'X': X}, 
                                    outputs={'logits': logits_op})
            # with end here


def main():
    parser = argparse.ArgumentParser('Train AlexNet on CIFAR-10')
    parser.add_argument('--dataset-dir', type=str, default='cifar-10-batches-py')
    parser.add_argument('--print-every', type=int, default=100)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--log-dir', type=str, default='log')
    parser.add_argument('--save-dir', type=str, default='model')
    parser.add_argument('--lrn', dest='lrn', action='store_true')
    parser.add_argument('--bn', dest='lrn', action='store_false')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--augment', dest='augment', action='store_true')
    parser.add_argument('--resume', dest='resume', action='store_true')
    parser.add_argument('--full-model', dest='full_model', action='store_true')
    parser.add_argument('--trunc-model', dest='full_model', action='store_false')
    parser.set_defaults(lrn=False, resume=False, augment=False, full_model=True)
    args = parser.parse_args()
    print(args)

    if not os.path.exists(args.dataset_dir):
        raise FileNotFoundError('Dataset does not exist in path: {}'.format(args.dataset_dir))
   
    if not os.path.exists(args.log_dir):
        os.mkdir(args.log_dir)
        print('Create log dir: {}'.format(args.log_dir))

    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
        print('Create save dir: {}'.format(args.save_dir))

    train(args)

if __name__ == '__main__':
    main()