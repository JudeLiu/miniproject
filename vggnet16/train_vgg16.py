import tensorflow as tf
import argparse
import os
import time
import numpy as np

from utils import rm_dir
from vgg16_trainable import VGG16, VGG16OneFC
from data_preprocess import load_data, prepare_dataset

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def check_accuracy(sess, dset, X, Y, correct_pred):
    num_correct, num_samples = 0, 0
    for x_batch, y_batch in dset:
        feed_dict = {X: x_batch, Y: y_batch}
        y_correct = correct_pred.eval(feed_dict=feed_dict)
        num_correct += y_correct.sum()
        num_samples += y_correct.shape[0]
        # logits_np = sess.run(logits, feed_dict=feed_dict)
        # y_pred = logits_np.argmax(axis=1)
        # num_samples += x_batch.shape[0]
        # num_correct += (y_pred == y_batch).sum()
    acc = float(num_correct) / num_samples
    
    return acc, num_correct, num_samples

performance = {4096: 71.5, 3072: 69.30, 2048: 68.20}

def build_model(X, Y, train_mode_ph, lr, **kwargs):
    full = kwargs.get('full')
    fc_feat_size = kwargs.get('feat_size')
    vgg16_npy_path = kwargs.get('vgg16_npy_path', 'vgg16.npy')
    global_step = kwargs.get('global_step', None)
    image_size = kwargs.get('image_size')
    # mean_image = kwargs.get('mean_image')

    # model = VGG16OneFC(vgg16_npy_path=vgg16_npy_path, num_classes=10)
    vgg = VGG16 if full else VGG16OneFC
    model = vgg(vgg16_npy_path=vgg16_npy_path, num_classes=10, fc_feat_size=fc_feat_size, global_step=global_step)
    model.build(X, image_size=image_size, train_mode=None)
    global_step = model.global_step
    logits_op = model.logits
    loss_op = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=Y, logits=logits_op)
    loss_op = tf.reduce_mean(loss_op, name='loss')
    correct_pred = tf.equal(tf.cast(tf.argmax(logits_op, axis=1), tf.int32), Y)
    acc_op = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='train_acc')

    if full:
        fc6_variables = tf.contrib.framework.get_variables('fc6/')
        fc7_variables = tf.contrib.framework.get_variables('fc7/')
        fc8_variables = tf.contrib.framework.get_variables('fc8/')
    else:
        fc8_variables = tf.contrib.framework.get_variables('logits/')
    var_list = [fc6_variables, fc7_variables, fc8_variables] if full else [fc8_variables]
    fc8_train_op = tf.train.AdamOptimizer(lr)\
                        .minimize(loss_op, global_step=global_step, var_list=var_list)
    full_train_op = tf.train.AdamOptimizer(lr).minimize(loss_op, global_step=global_step)

    return model, logits_op, loss_op, acc_op, correct_pred, fc8_train_op, full_train_op

def train(args):
    if args.standard_config:
        args.small = True
        args.feat = 512
        args.epoch1 = 5
        args.epoch2 = 30

    images, labels = load_data(train=True)
    train_dset, val_dset = prepare_dataset(images, labels, train=True, augment=args.augment)

    learning_rate = [1e-5]

    if args.resume or args.eval:
        assert args.model_dir is not None
        vgg16_npy_path = args.model_dir
    else:
        vgg16_npy_path = 'vgg16.npy'

    for lr in learning_rate:
        tf.reset_default_graph()
        graph = tf.Graph()

        with graph.as_default():
            # input placeholder
            input_shape = (32, 32) if not args.augment else (24, 24)
            X = tf.placeholder(tf.float32, [None, *input_shape, 3], name='images_ph')
            Y = tf.placeholder(tf.int32, [None], name='labels_ph')
            train_mode_ph = tf.placeholder(tf.bool, name='train_mode_ph')
            image_size = (24, 24) if args.augment else (32, 32)
            model, _, loss_op, acc_op, correct_pred, fc8_train_op, full_train_op = \
                build_model(X, Y, train_mode_ph, lr, 
                            image_size=image_size, #mean_image=mean_image,
                            feat_size=args.feat, vgg16_npy_path=vgg16_npy_path,
                            full=args.full)
            global_step = model.global_step
            print('npy path: {}'.format(vgg16_npy_path))

            middle_name = 'feat-{}'.format(args.feat)
            # add variables to summary
            log_dir = os.path.join(args.log_dir, middle_name, '')
            if not args.eval:
                rm_dir(log_dir)

            tf.summary.scalar('loss', loss_op)
            tf.summary.scalar('train_acc', acc_op)
            tf.summary.histogram('loss', loss_op)
            summary_op = tf.summary.merge_all()
            summary_writer = tf.summary.FileWriter(log_dir, graph)

            # model saver
            if not args.eval:
                save_dir = os.path.join(args.save_dir, middle_name, '')
            else:
                save_dir = args.save_dir
            # print('save at {}'.format(save_dir))
            # saver = tf.train.Saver(filename=save_dir)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
                print('Create save dir: ', save_dir)

            # run training session
            config = tf.ConfigProto(log_device_placement=False)
            with tf.Session(graph=graph, config=config) as sess:
                sess.run(tf.global_variables_initializer())
                # if args.eval:
                #     print('Resume from {}'.format(args.save_dir))
                #     saver.restore(sess, tf.train.latest_checkpoint(save_dir))
                print('lr={}, step: {}'.format(lr, global_step.eval())) 

                if not args.eval:
                    print('Train fc layers')
                    for e in range(args.epoch1):
                        print('Epoch {}'.format(e))
                        for x_batch, y_batch in train_dset:
                            _, acc, summary = sess.run([fc8_train_op, acc_op, summary_op], 
                                                feed_dict={X: x_batch, 
                                                            Y: y_batch,
                                                            train_mode_ph: True})
                            summary_writer.add_summary(summary, global_step.eval())
                        # saver.save(sess, save_dir, global_step=global_step)
                    if args.epoch1 > 0 and args.epoch2 == 0: 
                        model.save_npy(sess, npy_path=os.path.join(save_dir, 'vgg16-save-{}.npy'.format(int(time.time()))))
                        acc, num_correct, num_samples = check_accuracy(sess, val_dset, X, Y, correct_pred)
                        print('step:{}, val acc: {:.2%} ({}/{})'.format(global_step.eval (), acc, num_correct, num_samples))

                    print('Finetune conv layers')
                    for e in range(args.epoch2):
                        print('Epoch ', e)
                        for x_batch, y_batch in train_dset:
                            _, acc, summary = sess.run([full_train_op, acc_op, summary_op],
                                                        feed_dict={X: x_batch,
                                                                    Y: y_batch,
                                                                    train_mode_ph: True})
                            summary_writer.add_summary(summary, global_step.eval())
                    if args.epoch2 > 0:
                        model.save_npy(sess, npy_path=os.path.join(save_dir, 'vgg16-save-{}.npy'.format(int(time.time()))))        
                        acc, num_correct, num_samples = check_accuracy(sess, val_dset, X, Y, correct_pred)
                        print('step: {}, val acc:{:.2%}, ({}/{})'.format(global_step.eval(), acc, num_correct, num_samples))
                else:
                    images, labels = load_data(False)
                    test_dset = prepare_dataset(images, labels, train=False, 
                                                mean=train_dset.mean, std=train_dset.std,
                                                augment=args.augment)
                    acc, num_correct, num_samples = check_accuracy(sess, test_dset, X, Y, correct_pred)
                    print('step:{}, test acc: {:.2%} ({}/{})'.format(global_step.eval(), acc, num_correct, num_samples))
            # with end here

def test(args):
    images, labels = load_data(train=False)
    test_dset = prepare_dataset(images, labels, train=False, augment=args.augment)
    vgg16_npy_path = args.model_dir

    # input placeholder
    input_shape = (32, 32) if not args.augment else (24, 24)
    X = tf.placeholder(tf.float32, [None, *input_shape, 3], name='images_ph')
    Y = tf.placeholder(tf.int32, [None], name='labels_ph')
    vgg = VGG16 if args.full else VGG16OneFC
    model = vgg(vgg16_npy_path=vgg16_npy_path, num_classes=10, fc_feat_size=args.feat)
    model.build(X, train_mode=None, image_size=input_shape)
    logits_op = model.logits
    correct_pred = tf.equal(tf.cast(tf.argmax(logits_op, axis=1), tf.int32), Y)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        acc, num_correct, num_samples = check_accuracy(sess, test_dset, X, Y, correct_pred)
        print('step: ', sess.run(model.global_step))
        print('test acc: {:.2%} ({}/{})'.format(acc, num_correct, num_samples))

def compute_feature(dset, X, logits_op, feature_map, name):
    feats = {'feat':[], 'pred':[], 'gt': []}

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for x_batch, y_batch in dset:
            feature, logits = sess.run([feature_map, logits_op], feed_dict={X:x_batch})
            pred_y = np.argmax(logits, axis=1)
            # print(feature.shape)
            feats['feat'].append(feature)
            feats['pred'].append(pred_y)
            feats['gt'].append(y_batch)
 
        to_list = lambda x : np.concatenate(x).tolist()
        feats['feat'] = to_list(feats['feat'])
        feats['pred'] = to_list(feats['pred'])
        feats['gt'] = to_list(feats['gt'])
        np.save('feat_{}.npy'.format(name), feats)

def get_feature(args):
    images, labels = load_data(train=True)
    train_dset, val_dset = prepare_dataset(images, labels, train=True)
    # images, labels = load_data(train=False)
    # test_dset = prepare_dataset(images, labels, train=False)
    vgg16_npy_path = args.model_dir

    X = tf.placeholder(tf.float32, [None, 32, 32, 3], name='images_ph')
    # Y = tf.placeholder(tf.int32, [None], name='labels_ph')
    model = VGG16OneFC(vgg16_npy_path=vgg16_npy_path, num_classes=10, fc_feat_size=args.feat)
    model.build(X, image_size=args.image_size)
    feature_map = model.pool5
    feature_map = tf.reshape(feature_map, [-1, 512])
    logits_op = model.logits

    compute_feature(train_dset, X, logits_op, feature_map, 'train')
    compute_feature(val_dset, X, logits_op, feature_map, 'val')
    images, labels = load_data(train=False)
    test_dset = prepare_dataset(images, labels, train=False)
    compute_feature(test_dset, X, logits_op, feature_map, 'test')

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Train AlexNet on CIFAR-10')
    parser.add_argument('--epoch1', type=int, default=0)
    parser.add_argument('--epoch2', type=int, default=1)
    parser.add_argument('--log-dir', type=str, default='log')
    parser.add_argument('--save-dir', type=str, default='model')
    parser.add_argument('--npy-path', type=str, default='vgg16.npy')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--augment', dest='augment', action='store_true')
    parser.add_argument('--resume', dest='resume', action='store_true')
    parser.add_argument('--eval', dest='eval', action='store_true')
    parser.add_argument('--init', type=str, default='he')
    parser.add_argument('--feat', type=int, default=4096)
    parser.add_argument('--model-dir')
    parser.add_argument('--full', dest='full', action='store_true')
    parser.add_argument('--small', dest='full', action='store_false')
    parser.add_argument('--compute-feature', dest='compute_feature', action='store_true')
    parser.add_argument('--standard', dest='standard_config', action='store_true')
    parser.set_defaults(resume=False, augment=False, eval=False,
                        compute_feature=False, standard_config=False)
    args = parser.parse_args()
    print(args)

    if args.compute_feature:
        get_feature(args)
    else:
        if not args.eval:
            train(args)
        else:
            test(args)
