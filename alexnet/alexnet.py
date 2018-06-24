import tensorflow as tf

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
    def __init__(self, lrn, initializer):
        super(AlexNet, self).__init__()
        # initializer = tf.variance_scaling_initializer(scale=2)
        if initializer == 'xavier':
            weight_initializer = tf.contrib.layers.xavier_initializer()
        elif initializer == 'he':
            weight_initializer = tf.keras.initializers.he_normal()
        elif initializer == 'normal':
            weight_initializer = tf.random_normal_initializer(mean=0, stddev=.01)
        else:
            raise ValueError('%s Not defined' % initializer)
        
        first_and_second_layer = ConvLRNPoolLayer if lrn else ConvBNPoolLayer
        # self.layer1 = first_and_second_layer(filters=96, kernel_size=11, strides=4)
        self.layer1 = first_and_second_layer(filters=96, kernel_size=5, strides=1)
        self.layer2 = first_and_second_layer(filters=256, kernel_size=5, strides=1)
        self.layer3 = tf.layers.Conv2D(filters=384, kernel_size=3, strides=1, padding='same',
                                        kernel_initializer=weight_initializer,
                                        kernel_regularizer=tf.nn.l2_loss)
        self.layer4 = tf.layers.Conv2D(filters=384, kernel_size=3, strides=1, padding='same',
                                        kernel_initializer=weight_initializer,
                                        kernel_regularizer=tf.nn.l2_loss)
        self.layer5 = ConvPoolLayer(256, kernel_size=3, strides=1)
        self.flatten = tf.layers.Flatten()
        self.fc1 = tf.layers.Dense(1024, activation=tf.nn.relu,
                                    kernel_initializer=weight_initializer,
                                    kernel_regularizer=tf.nn.l2_loss)
        self.fc2 = tf.layers.Dense(1024, activation=tf.nn.relu,
                                    kernel_initializer=weight_initializer,
                                    kernel_regularizer=tf.nn.l2_loss)
        # linear output for softmax
        self.fc3 = tf.layers.Dense(10, 
                                    kernel_initializer=weight_initializer,
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

class AlexNetTruncated(tf.keras.Model):
    def __init__(self, lrn=True):
        super(AlexNetTruncated, self).__init__()
        first_and_second_layer = ConvLRNPoolLayer if lrn else ConvBNPoolLayer
        self.layer1 = first_and_second_layer(32, 5, 1, 'same')
        self.layer2 = first_and_second_layer(32, 5, 1, 'same')
        self.layer3 = ConvPoolLayer(64, 5, 1, 'same')
        self.flatten = tf.layers.Flatten()
        self.fc1 = tf.layers.Dense(1024, activation=tf.nn.relu,
                                    kernel_initializer=tf.contrib.layers.xavier_initializer(),)
        self.fc2 = tf.layers.Dense(10, 
                                    kernel_initializer=tf.contrib.layers.xavier_initializer(),)

    def __call__(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x
