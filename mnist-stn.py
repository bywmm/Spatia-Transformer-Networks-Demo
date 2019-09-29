import tensorflow as tf
import numpy as np
from stn import *
from datetime import datetime
from PIL import Image
from matplotlib import pyplot as plt

def load_mnist():
    mnist = np.load('MNIST_data/mnist.npz')
    x_train, y_train = mnist['x_train'], mnist['y_train']
    x_test, y_test = mnist['x_test'], mnist['y_test']
    y_train = np.eye(10)[y_train]
    y_test = np.eye(10)[y_test]
    mnist.close()
    return x_train, y_train, x_test, y_test

x_train, y_train, x_test, y_test = load_mnist()

# 为输入值设立placeholder
with tf.name_scope('input'):
    x = tf.placeholder(tf.float32, [None, 28, 28])
    y = tf.placeholder(tf.float32, [None, 10])

# [B, H, W, C]
x_image = tf.reshape(x, [-1, 28, 28, 1])


# init of W and b
def weight_variable(shape):
    # initial = tf.truncated_normal(shape, stddev=0.1)
    initial = tf.zeros(shape)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


# Conv2D and Pooling
def conv2D(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')

# stn
with tf.name_scope('stn_loc'):
    with tf.name_scope('loc_conv1'):
        W_loc_conv1 = weight_variable([7, 7, 1, 8])
        tf.summary.histogram('W_loc_conv1', W_loc_conv1)
        b_loc_conv1 = bias_variable([8])
        loc_conv1 = conv2D(x_image, W_loc_conv1)
        h_loc_conv1 = tf.nn.relu(loc_conv1 + b_loc_conv1)
        h_loc_pool1 = max_pool_2x2(h_loc_conv1)

    with tf.name_scope('loc_conv2'):
        W_loc_conv2 = weight_variable([5, 5, 8, 10])
        # tf.summary.histogram('W_loc_conv2', W_loc_conv2)
        b_loc_conv2 = bias_variable([10])
        loc_conv2 = conv2D(h_loc_pool1, W_loc_conv2)
        h_loc_conv2 = tf.nn.relu(loc_conv2 + b_loc_conv2)
        h_loc_pool2 = max_pool_2x2(h_loc_conv2)
        # output size: [7, 7, 10]

    with tf.name_scope('loc_fc1'):
        W_loc_fc1 = weight_variable([7 * 7 * 10, 32])
        # tf.summary.histogram('W_loc_fc1', W_loc_fc1)
        b_loc_fc1 = bias_variable([32])
        h_loc_pool2_flat = tf.reshape(h_loc_pool2, [-1, 7*7*10])
        h_loc_fc1 = tf.nn.relu(tf.matmul(h_loc_pool2_flat, W_loc_fc1) + b_loc_fc1)

    with tf.name_scope('loc_fc2'):
        W_loc_fc2 = weight_variable([32, 6])
        # tf.summary.histogram('W_loc_fc2', W_loc_fc2)
        b_loc_fc2 = bias_variable([6])
        theta = tf.matmul(h_loc_fc1, W_loc_fc2) + b_loc_fc2
        print(theta.shape)

    with tf.name_scope('transformer'):
        stn_image = spatial_transformer_network(x_image, theta)
        # tf.summary.histogram('histogram_theta', theta)

print(stn_image.shape, x_image.shape)

# 卷积核为[5, 5, 1, 32] patch-size：5*5, 输出32个特征值
with tf.name_scope('recoginizer'):
    with tf.name_scope('conv1'):
        W_conv1 = weight_variable([5, 5, 1, 32])
        tf.summary.histogram('W_conv1', W_conv1)
        b_conv1 = bias_variable([32])
        # conv1 = conv2D(x_image, W_conv1)
        conv1 = conv2D(stn_image, W_conv1)
        h_conv1 = tf.nn.relu(conv1 + b_conv1)
        # output size: [28, 28,  32]
        h_pool1 = max_pool_2x2(h_conv1)
        # output size: [14, 14, 32]

    with tf.name_scope('conv2'):
        W_conv2 = weight_variable([5, 5, 32, 64])
        # tf.summary.histogram('W_conv2', W_conv2)
        b_conv2 = bias_variable([64])
        conv2 = conv2D(h_pool1, W_conv2)
        h_conv2 = tf.nn.relu(conv2 + b_conv2)
        # output size: [14, 14, 64]
        h_pool2 = max_pool_2x2(h_conv2)
        # output size: [7, 7, 64]

    with tf.name_scope('fc1'):
        W_fc1 = weight_variable([7 * 7 * 64, 1024])
        # tf.summary.histogram('W_fc1', W_fc1)
        b_fc1 = bias_variable([1024])
        h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
        # dropout: 训练时打开，测试时关闭
        keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    with tf.name_scope('fc2'):
        W_fc2 = weight_variable([1024, 10])
        # tf.summary.histogram('W_fc2', W_fc2)
        b_fc2 = bias_variable([10])
        # 最后一层： readout layer
        y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

# training
with tf.Session() as sess:
    # 交叉熵损失
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=y_conv, labels=y))
    tf.summary.scalar('cross_entropy', cross_entropy)
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', accuracy)
    merged = tf.summary.merge_all()
    TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())
    writer = tf.summary.FileWriter('log/mnist/' + TIMESTAMP, sess.graph)

    # 初始化变量
    sess.run(tf.global_variables_initializer())

    for epoch in range(100):
        for i in range(1200):
            x_batch = x_train[i*50:(i+1)*50]
            y_batch = y_train[i*50:(i+1)*50]
            train_step.run(feed_dict={x:x_batch, y:y_batch, keep_prob:0.5})
            if i % 200 == 0:
                # evaluation without dropout layer
                summary, acc, stn_image_out, x_image_out = sess.run([merged, accuracy, stn_image, x_image], feed_dict={x: x_batch, y: y_batch, keep_prob: 1.0})
                print("epoch %d, training accuracy %g" % (epoch, acc))
                # tf.summary.scalar('train_acc', acc)
                writer.add_summary(summary, epoch*1200+i)

                # plot the result of mnist-stn
                if(i % 1200 == 0 and epoch % 5 == 0):
                    if (epoch == 0):
                        plt.figure()
                        plt.subplot(2, 2, 4)
                        plt.imshow(x_image_out[4].squeeze())
                        plt.subplot(2, 2, 1)
                        plt.imshow(x_image_out[1].squeeze())
                        plt.subplot(2, 2, 2)
                        plt.imshow(x_image_out[2].squeeze())
                        plt.subplot(2, 2, 3)
                        plt.imshow(x_image_out[3].squeeze())
                        plt.show()

                    plt.figure()
                    plt.subplot(2, 2, 4)
                    plt.imshow(stn_image_out[4].squeeze())
                    plt.subplot(2, 2, 1)
                    plt.imshow(stn_image_out[1].squeeze())
                    plt.subplot(2, 2, 2)
                    plt.imshow(stn_image_out[2].squeeze())
                    plt.subplot(2, 2, 3)
                    plt.imshow(stn_image_out[3].squeeze())
                    plt.show()


    print("test accuracy %g" % accuracy.eval(feed_dict={
        x: x_test, y:y_test, keep_prob: 1.0}))

    writer.close()

