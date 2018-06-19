import tensorflow as tf
import argparse
import numpy as np
import os
from data_preprocess import load_data

os.environ["CUDA_VISIBLE_DEVICES"]="0"

class ConvLRNPoolLayer(tf.keras.Model):
    def __init__(self, filters, kernel_size, strides, padding='same'):
        super(ConvLRNPoolLayer, self).__init__()
        self.conv = tf.layers.Conv2D(filters=filters, kernel_size=kernel_size,
                                        padding='same', strides=strides, 
                                        activation=tf.nn.relu, use_bias=True,
                                        kernel_initializer=tf.contrib.layers.xavier_initializer())
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


class ConvPoolLayer(tf.keras.Model):
    def __init__(self, filters, kernel_size=3, strides=1, padding='same'):
        super(ConvPoolLayer, self).__init__()
        self.conv = tf.layers.Conv2D(filters=filters, kernel_size=kernel_size, 
                                        padding=padding, strides=strides,
                                        activation=tf.nn.relu, use_bias=True,
                                        kernel_initializer=tf.contrib.layers.xavier_initializer())
        self.maxpooling = tf.layers.MaxPooling2D(pool_size=3, strides=2, padding='same')

    def __call__(self, x):
        x = self.conv(x)
        x = self.maxpooling(x)
        return x


class AlexNet(tf.keras.Model):
    def __init__(self):
        super(AlexNet, self).__init__()
        # initializer = tf.variance_scaling_initializer(scale=2)
        self.layer1 = ConvLRNPoolLayer(filters=96, kernel_size=11, strides=4)   
        self.layer2 = ConvLRNPoolLayer(filters=256, kernel_size=5, strides=1)
        self.layer3 = tf.layers.Conv2D(filters=384, kernel_size=3, strides=1, padding='same',
                                        kernel_initializer=tf.contrib.layers.xavier_initializer())
        self.layer4 = tf.layers.Conv2D(filters=384, kernel_size=3, strides=1, padding='same',
                                        kernel_initializer=tf.contrib.layers.xavier_initializer())
        self.layer5 = ConvPoolLayer(256, kernel_size=3, strides=1)
        self.flatten = tf.layers.Flatten()
        self.fc1 = tf.layers.Dense(4096, kernel_initializer=tf.contrib.layers.xavier_initializer())
        self.fc2 = tf.layers.Dense(10, kernel_initializer=tf.contrib.layers.xavier_initializer())

    def __call__(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

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


def prepare_dataset(num_training=49000, num_validation=1000, num_test=10000):
    images, labels = load_data(train=True)
    mask = range(num_training, num_training + num_validation)
    X_val = images[mask]
    y_val = labels[mask]
    mask = range(num_training)
    X_train = images[mask]
    y_train = labels[mask]

    X_test, y_test = load_data(train=False)
    train_dset = Dataset(X_train, y_train, batch_size=64, shuffle=True)
    val_dset = Dataset(X_val, y_val, batch_size=64, shuffle=False)
    test_dset = Dataset(X_test, y_test, batch_size=64)
    return [train_dset, val_dset, test_dset]


def check_accuracy(sess, dset, X, scores):
    num_correct, num_samples = 0, 0
    for x_batch, y_batch in dset:
        feed_dict = {X: x_batch}
        scores_np = sess.run(scores, feed_dict=feed_dict)
        y_pred = scores_np.argmax(axis=1)
        num_samples += x_batch.shape[0]
        num_correct += (y_pred == y_batch).sum()
    acc = float(num_correct) / num_samples
    
    return acc, num_correct, num_samples


def train(args):
    train_dset, val_dset, test_dset = prepare_dataset()
    # learning_rate = 1e-6*(90 + (6 * np.random.random(5) - 2))
    learning_rate = [9.051e-5]
    
    for lr in learning_rate:
        tf.reset_default_graph()
        X = tf.placeholder(tf.float32, [None, 32, 32, 3])
        Y = tf.placeholder(tf.int32, [None])

        model = AlexNet()
        scores = model(X)
        loss_op = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=Y, logits=scores)
        loss_op = tf.reduce_mean(loss_op)
        correct_pred = tf.equal(tf.cast(tf.argmax(scores, axis=1), tf.int32), Y)
        acc_op = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    
        optim = tf.train.AdamOptimizer(lr)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = optim.minimize(loss_op)

        tf.summary.scalar('loss', loss_op)
        tf.summary.scalar('train_acc', acc_op)
        tf.summary.histogram('loss', loss_op)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            print('lr={}'.format(lr))
            cnt = 0
            log_dir = '{}/lr-{:.8f}'.format(args.log_dir, lr)
            summary_writer = tf.summary.FileWriter(log_dir, sess.graph)

            for e in range(args.epochs):
                for i, (x_batch, y_batch) in enumerate(train_dset):
                    summary_op = tf.summary.merge_all()
                    loss, _, acc, summary = sess.run([loss_op, train_op, acc_op, summary_op], feed_dict={X: x_batch, Y: y_batch})
                    # acc, num_correct, num_samples = check_accuracy(sess, val_dset, X, scores)
                    
                    summary_writer.add_summary(summary, cnt)
                    cnt += 1
                    if i % args.print_every == 0:
                        print('loss:{:.4f}, acc:{:.2%}'.format(loss, acc))
                        # print('loss:{:.4f}, acc:{:.2%}, {} / {} correct'.format(loss, acc, num_correct, num_samples))
            tf.saved_model.simple_save(sess, log_dir, inputs={'x':X, 'y':Y}, outputs={'scores':scores})

def main():
    parser = argparse.ArgumentParser('Train AlexNet on CIFAR-10')
    parser.add_argument('--dataset-dir', type=str, default='cifar-10-batches-py')
    parser.add_argument('--print-every', type=int, default=100)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--log-dir', type=str, default='train_log')
    args = parser.parse_args()
    print(args)

    train(args)

if __name__ == '__main__':
    main()
