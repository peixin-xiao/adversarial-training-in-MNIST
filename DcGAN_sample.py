
# from __future__ import division, print_function, absolute_import

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# 导入MNIST数据集
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

# 训练参数
num_steps = 2000
batch_size = 32

# 网络参数
image_dim = 784  # 28*28 pixels * 1 channel
gen_hidden_dim = 256
disc_hidden_dim = 256
noise_dim = 200  # 噪声数据


# 定义生成网络
# 输入：噪声；输出：图片
def generator(x, reuse=False):
    with tf.variable_scope('Generator', reuse=reuse):
        x = tf.layers.dense(x, units=6 * 6 * 128)
        x = tf.nn.tanh(x)
        # 将图片变换为4维张量: (batch, height, width, channels)
        # 新的形状: (batch, 6, 6, 128)
        x = tf.reshape(x, shape=[-1, 6, 6, 128])
        # 反卷积, 图片的形状: (batch, 14, 14, 64)
        x = tf.layers.conv2d_transpose(x, 64, 4, strides=2)
        # 反卷积, 图片的形状: (batch, 28, 28, 1)
        x = tf.layers.conv2d_transpose(x, 1, 2, strides=2)
        # 使用sigmoid激活函数激活
        x = tf.nn.sigmoid(x)
        return x


# 定义判别网络
# 输入：图片, 输出: 预测的真假图片
def discriminator(x, reuse=False):
    with tf.variable_scope('Discriminator', reuse=reuse):
        # 用经典的CNN去分类
        x = tf.layers.conv2d(x, 64, 5)
        x = tf.nn.tanh(x)
        x = tf.layers.average_pooling2d(x, 2, 2)
        x = tf.layers.conv2d(x, 128, 5)
        x = tf.nn.tanh(x)
        x = tf.layers.average_pooling2d(x, 2, 2)
        x = tf.contrib.layers.flatten(x)
        x = tf.layers.dense(x, 1024)
        x = tf.nn.tanh(x)
        # 输出为2类，真假图片
        x = tf.layers.dense(x, 2)
    return x

# 创建GAN网络
# 网络输入
noise_input = tf.placeholder(tf.float32, shape=[None, noise_dim])
real_image_input = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])

#### 创建生成网络
gen_sample = generator(noise_input)

# 创建两个判别网络，一个用于真实图片，一个用于生成样本
disc_real = discriminator(real_image_input)
disc_fake = discriminator(gen_sample, reuse=True)
disc_concat = tf.concat([disc_real, disc_fake], axis=0)
stacked_gan = discriminator(gen_sample, reuse=True)

# 创建目标
disc_target = tf.placeholder(tf.int32, shape=[None])
gen_target = tf.placeholder(tf.int32, shape=[None])

# 定义损失函数
disc_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
    logits=disc_concat, labels=disc_target))
gen_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
    logits=stacked_gan, labels=gen_target))

# 采用AdamOptimizer优化
optimizer_gen = tf.train.AdamOptimizer(learning_rate=0.002)
optimizer_disc = tf.train.AdamOptimizer(learning_rate=0.0008)

# 网络变量
gen_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Generator')
disc_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Discriminator')

# 最小化损失函数
train_gen = optimizer_gen.minimize(gen_loss, var_list=gen_vars)
train_disc = optimizer_disc.minimize(disc_loss, var_list=disc_vars)

# 初始化变量
init = tf.global_variables_initializer()

# 开始训练
with tf.Session() as sess:
    sess.run(init)

    for i in range(1, num_steps+1):
        batch_x, _ = mnist.train.next_batch(batch_size)
        batch_x = np.reshape(batch_x, newshape=[-1, 28, 28, 1])
        z = np.random.uniform(-1., 1., size=[batch_size, noise_dim])

        # 准备目标 (真图片: 1, 假图片: 0)
        # 前半部分数据为真图片，后半部分数据为假图片
        batch_disc_y = np.concatenate(
            [np.ones([batch_size]), np.zeros([batch_size])], axis=0)
        # 生成器尝试去欺骗判别器, 因此目标是1.
        batch_gen_y = np.ones([batch_size])

        # 训练
        feed_dict = {real_image_input: batch_x, noise_input: z,
                     disc_target: batch_disc_y, gen_target: batch_gen_y}
        _, _, gl, dl = sess.run([train_gen, train_disc, gen_loss, disc_loss],
                                feed_dict=feed_dict)
        if i % 100 == 0 or i == 1:
            print('Step %i: Generator Loss: %f, Discriminator Loss: %f' % (i, gl, dl))

    # 用生成器生成的网络来自噪声
    f, a = plt.subplots(4, 10, figsize=(10, 4))
    for i in range(10):
        # 噪声输入
        z = np.random.uniform(-1., 1., size=[4, noise_dim])
        g = sess.run(gen_sample, feed_dict={noise_input: z})
        for j in range(4):
            # 将噪声生成的图片拓展成三通道用于展示.
            img = np.reshape(np.repeat(g[j][:, :, np.newaxis], 3, axis=2),
                             newshape=(28, 28, 3))
            a[j][i].imshow(img)

    f.show()
    plt.draw()
    plt.waitforbuttonpress()


