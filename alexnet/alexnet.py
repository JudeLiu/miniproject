import tensorflow as tf
import argparse
import numpy as np
import os
import shutil
from data_preprocess import load_data, prepare_dataset

os.environ["CUDA_VISIBLE_DEVICES"]="0"

class ConvLRNPoolLayer(tf.keras.Model):
    def __init__(self, filters, kernel_size, strides, padding='same'):
        super(ConvLRNPoolLayer, self).__init__()
        self.conv = tf.layers.Conv2D(filters=filters, kernel_size=kernel_size,
                                        padding='same', strides=strides, 
                                        activation=tf.nn.relu, use_bias=True,
                                        kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                        kernel_regularizer=tf.nn.l2_loss)
        self.maxpooling = tf.layers.MaxPooling2D(pool_size=3, strides=2, padding='same')
    
    def __call__(self, x):
        x = self.conv(x)
        x = tf.nn.local_response_normalization(x, 
                                                depth_radius=5, 
                                                bias=2, 
                                                alpha=1e-4, 
                                                beta=0.75)
        x = self.maxpooling(x)
        return x                                             

class ConvBNPoolLayer(tf.keras.Model):
    def __init__(self, filters, kernel_size, strides, padding='same'):
        super(ConvBNPoolLayer, self).__init__()
        self.conv = tf.layers.Conv2D(filters=filters, kernel_size=kernel_size,
                                        padding='same', strides=strides, 
                                        activation=tf.nn.relu, use_bias=True,
                                        kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                        kernel_regularizer=tf.nn.l2_loss)
        self.bn = tf.layers.BatchNormalization()                                        
        self.maxpooling = tf.layers.MaxPooling2D(pool_size=3, strides=2, padding='same')

    def __call__(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.maxpooling(x)
        return x

class ConvPoolLayer(tf.keras.Model):
    def __init__(self, filters, kernel_size=3, strides=1, padding='same'):
        super(ConvPoolLayer, self).__init__()
        self.conv = tf.layers.Conv2D(filters=filters, kernel_size=kernel_size, 
                                        padding=padding, strides=strides,
                                        activation=tf.nn.relu, use_bias=True,
                                        kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                        kernel_regularizer=tf.nn.l2_loss)
        self.maxpooling = tf.layers.MaxPooling2D(pool_size=3, strides=2, padding='same')

    def __call__(self, x):
        x = self.conv(x)
        x = self.maxpooling(x)
        return x


class AlexNet(tf.keras.Model):
    def __init__(self, lrn=True):
        super(AlexNet, self).__init__()
        # initializer = tf.variance_scaling_initializer(scale=2)
        first_and_second_layer = ConvLRNPoolLayer if lrn else ConvBNPoolLayer
        # self.layer1 = first_and_second_layer(filters=96, kernel_size=11, strides=4)
        self.layer1 = first_and_second_layer(filters=96, kernel_size=5, strides=1)
        self.layer2 = first_and_second_layer(filters=256, kernel_size=5, strides=1)
        self.layer3 = tf.layers.Conv2D(filters=384, kernel_size=3, strides=1, padding='same',
                                        kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                        kernel_regularizer=tf.nn.l2_loss)
        self.layer4 = tf.layers.Conv2D(filters=384, kernel_size=3, strides=1, padding='same',
                                        kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                        kernel_regularizer=tf.nn.l2_loss)
        self.layer5 = ConvPoolLayer(256, kernel_size=3, strides=1)
        self.flatten = tf.layers.Flatten()
        self.fc1 = tf.layers.Dense(1024, activation=tf.nn.relu,
                                    kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                    kernel_regularizer=tf.nn.l2_loss)
        self.fc2 = tf.layers.Dense(1024, activation=tf.nn.relu,
                                    kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                    kernel_regularizer=tf.nn.l2_loss)
        # linear output for softmax
        self.fc3 = tf.layers.Dense(10, 
                                    kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                    kernel_regularizer=tf.nn.l2_loss)

    def __call__(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

class AlexNetSimplified(tf.keras.Model):
    def __init__(self, lrn=True):
        super(AlexNetSimplified, self).__init__()
        first_and_second_layer = ConvLRNPoolLayer if lrn else ConvBNPoolLayer
        self.layer1 = first_and_second_layer(32, 5, 1, 'same')
        self.layer2 = first_and_second_layer(32, 5, 1, 'same')
        self.layer3 = ConvPoolLayer(64, 5, 1, 'same')
        self.flatten = tf.layers.Flatten()
        self.fc = tf.layers.Dense(10, kernel_initializer=tf.contrib.layers.xavier_initializer(),)
    
    def __call__(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

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

def train(args):
    images, labels = load_data(True)
    train_dset, val_dset = prepare_dataset(images, labels, True, augment=False)
    # images, labels = load_data(False)
    # test_dset = prepare_dataset(images, labels, False)

    # learning_rate = (1+.4*np.random.random(5))*1e-4
    learning_rate = [0.00012353763879248484]
    # learning_rate = [0.00012272]

    for lr in learning_rate:
        tf.reset_default_graph()
        graph = tf.Graph()
        
        with graph.as_default():
            if not args.resume:
                # build model, loss and optimizer
                # input placeholder
                X = tf.placeholder(tf.float32, [None, 32, 32, 3], name='images_ph')
                # X = tf.placeholder(tf.float32, [None, 24, 24, 3], name='images_ph')
                Y = tf.placeholder(tf.int32, [None], name='labels_ph')

                model = AlexNetSimplified(args.lrn)
                logits_op = model(X)
                loss_op = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=Y, logits=logits_op)
                loss_op = tf.reduce_mean(loss_op)
                correct_pred = tf.equal(tf.cast(tf.argmax(logits_op, axis=1), tf.int32), Y)
                acc_op = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
            
                optim = tf.train.AdamOptimizer(lr)
                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                with tf.control_dependencies(update_ops):
                    train_op = optim.minimize(loss_op)
            else:
                X = graph.get_tensor_by_name('images_ph:0')
                Y = graph.get_tensor_by_name('labels_ph:0')
                logits_op = graph.get_tensor_by_name('dense_2/BiasAdd:0')                    

            # add variables to summary
            log_dir = '{}/lr-{:.8f}'.format(args.log_dir, lr)
            if os.path.exists(log_dir):
                try:
                    shutil.rmtree(log_dir)
                    print('Remove existed dir: {}'.format(log_dir))
                except OSError as e:
                    print('Error: {} - {}.'.format(e.filename, e.strerror))
            tf.summary.scalar('loss', loss_op)
            tf.summary.scalar('train_acc', acc_op)
            tf.summary.histogram('loss', loss_op)

            # save builder
            save_dir = '{}/lr-{:.8f}'.format(args.save_dir, lr) if not args.resume else args.save_dir
            if os.path.exists(save_dir):
                try:
                    shutil.rmtree(save_dir)
                    print('Remove existed dir: {}'.format(save_dir))
                except OSError as e:
                    print('Error: {} - {}.'.format(e.filename, e.strerror))
            # builder = tf.saved_model.builder.SavedModelBuilder(save_dir)

            # run training session
            with tf.Session(graph=graph) as sess:
                # builder.add_meta_graph_and_variables(sess,
                #                            [tf.saved_model.tag_constants.TRAINING],
                #                            signature_def_map=)
                if not args.resume:
                    sess.run(tf.global_variables_initializer())
                else:
                    print('Resume from {}'.format(save_dir))
                    tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], save_dir)
                print('lr={}'.format(lr)) 
                cnt = 0
                summary_writer = tf.summary.FileWriter(log_dir, sess.graph)

                for e in range(args.epochs):
                    print('Epoch {}'.format(e))
                    for i, (x_batch, y_batch) in enumerate(train_dset):
                        summary_op = tf.summary.merge_all()
                        loss, _, acc, summary = sess.run([loss_op, train_op, acc_op, summary_op], feed_dict={X: x_batch, Y: y_batch})
                        # acc, num_correct, num_samples = check_accuracy(sess, val_dset, X, logits)
                        
                        summary_writer.add_summary(summary, cnt)
                        cnt += 1
                        if i % args.print_every == 0:
                            print('loss:{:.4f}, acc:{:.2%}'.format(loss, acc))
                            # print('loss:{:.4f}, acc:{:.2%}, {} / {} correct'.format(loss, acc, num_correct, num_samples))
                    acc, num_correct, num_samples = check_accuracy(sess, val_dset, X, logits_op)
                    print('val acc: {:.2%} ({}/{})'.format(acc, num_correct, num_samples))
            
                tf.saved_model.simple_save(sess, save_dir, 
                                    inputs={'X': X}, 
                                    outputs={'logits': logits_op})
            # with end here
            # builder.save()

def main():
    parser = argparse.ArgumentParser('Train AlexNet on CIFAR-10')
    parser.add_argument('--dataset-dir', type=str, default='cifar-10-batches-py')
    parser.add_argument('--print-every', type=int, default=100)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--log-dir', type=str, default='train_log')
    parser.add_argument('--save-dir', type=str, default='model')
    parser.add_argument('--lrn', dest='lrn', action='store_true')
    parser.add_argument('--bn', dest='lrn', action='store_false')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--resume', dest='resume', action='store_true')
    parser.set_defaults(lrn=False, resume=False)
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
