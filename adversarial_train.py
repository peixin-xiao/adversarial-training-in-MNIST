##
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os
# import numpy as np
# import matplotlib.pyplot as plt

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# 对标签进行One-Hot编码
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
# 参数设置
n_classes = 10  # 分类的类别
batch_size = 100  # batch的大小
train_save_path = "./train_results/model"
# 定义占位符
x = tf.placeholder('float', [None, 28 * 28])
y = tf.placeholder('float')
img = tf.placeholder('float', [batch_size, 28*28])
img_test = tf.placeholder('float', [10000, 28*28])
# 定义卷积层计算，步长为1，填充策略为SAME
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


# 定义池化层计算，步长为2，填充策略为SAME
def maxpool2d(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


# 定义网络结构及计算过程
def neural_network_model(data):
    # 图像大小 28*28 -- 14*14 -- 7*7
    # 采用字典定义网络结构
    with tf.name_scope(name='variable_for_disc'):

        # 卷积层1的权重，32个5×5卷积核
        w_conv1 = tf.Variable(tf.random_normal(shape=[5, 5, 1, 32]), name="w_conv1")
        # 卷积层2的权重，64个5×5卷积核
        w_conv2 = tf.Variable(tf.random_normal(shape=[5, 5, 32, 64]), name="w_conv2")
        # 全连接层，1024个神经元
        w_fc = tf.Variable(tf.random_normal(shape=[7 * 7 * 64, 1024]), name="w_fc")
        # 输出层，10个神经元
        w_out = tf.Variable(tf.random_normal(shape=[1024, n_classes]), name="w_out")


        # 卷积层1的偏置，共32个
        b_conv1 = tf.Variable(tf.random_normal([32]), name="b_conv1")
        # 卷积层2的偏置，共64个
        b_conv2 = tf.Variable(tf.random_normal([64]), name="b_conv2")
        # 全连接层的偏置，共1024个
        b_fc = tf.Variable(tf.random_normal([1024]), name="b_fc")
        # 输出层的偏置，共10个
        b_out = tf.Variable(tf.random_normal([n_classes]), name="b_out")

        # 数据维度转化
        data = tf.reshape(data, [-1, 28, 28, 1])

        # 每层计算过程
        # 卷积层1，卷积+池化
        conv1 = tf.nn.relu(conv2d(data, w_conv1) + b_conv1)
        conv1 = maxpool2d(conv1)

        # 卷积层2，卷积+池化
        conv2 = tf.nn.relu(conv2d(conv1, w_conv2) + b_conv2)
        conv2 = maxpool2d(conv2)

        # 全连接层
        fc = tf.reshape(conv2, [-1, 7 * 7 * 64])
        fc = tf.nn.relu(tf.matmul(fc, w_fc) + b_fc)

        # 输出层，无需经过Softmax，只需计算加权和
        output = tf.matmul(fc, w_out) + b_out

        return output


def add_noise(img):
    with tf.name_scope(name='variable_for_gen'):
        noise_variables = tf.Variable(tf.random_normal([1, 784]), name="noise_variables")
        noise_variables1 = tf.clip_by_value(noise_variables, -0.05, 0.05)
        img_size = img.get_shape()[0]
        img = tf.convert_to_tensor(img)
        img_out = tf.add(img[0, :], noise_variables1)
        img_out = tf.reshape(img_out, (1, 784))
        for i in range(1, img_size):
            img_single = (tf.add(img[i, :], noise_variables1))
            img_out = tf.concat([img_out, img_single], 0)
        img_out = tf.clip_by_value(img_out, 0, 1)
        intensity = tf.reduce_mean(tf.abs(noise_variables1))

        return img_out, intensity



img_with_noise, intensity = add_noise(img)
img_with_noise_larger, _ = add_noise(img_test)
prediction = neural_network_model(x)
prediction_with_noise = neural_network_model(img_with_noise)
test_for_prediction_with_noise = neural_network_model(img_with_noise_larger)
loss0 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
loss1 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction_with_noise, labels=y))
loss = loss0*loss1
value_of_jam = 1/loss1 + 50*intensity
vars0 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='variable_for_disc')
vars1 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='variable_for_gen')
optimizer0 = tf.train.AdamOptimizer().minimize(loss, var_list=vars0)
optimizer1 = tf.train.AdamOptimizer().minimize(value_of_jam, var_list=vars1)
hm_epochs = 10
saver = tf.train.Saver(var_list=vars0, max_to_keep=1)
with tf.Session() as sess:
    # 参数初始化
    sess.run(tf.global_variables_initializer())
    for epoch in range(hm_epochs):
        epoch_loss = 0  # 累计损失
        for _ in range(int(mnist.train.num_examples / batch_size)):
            # 提取数据
            epoch_x, epoch_y = mnist.train.next_batch(batch_size)
            # 喂数据
            _, _, c, _, noise_intensity = sess.run([optimizer0, optimizer1, loss, value_of_jam, intensity],
                                                   feed_dict={x: epoch_x, y: epoch_y, img:  epoch_x})
            epoch_loss = epoch_loss + c

        print('Epoch', epoch, 'completed out of', hm_epochs, 'loss', epoch_loss)
    # 计算分类正确率
    correct0 = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
    correct1 = tf.equal(tf.argmax(test_for_prediction_with_noise, 1), tf.argmax(y, 1))
    accuracy0 = tf.reduce_mean(tf.cast(correct0, 'float'))
    accuracy1 = tf.reduce_mean(tf.cast(correct1, 'float'))
    print('Accuracy0:', accuracy0.eval({x: mnist.test.images, y: mnist.test.labels}))
    print('Accuracy1:', accuracy1.eval({img_test: mnist.test.images, y: mnist.test.labels}))
    ##
    print("saving...")
    saver.save(sess, save_path=train_save_path)




print('hello world')





