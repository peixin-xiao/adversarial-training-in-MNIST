##
# import matplotlib.pyplot as plt
# import numpy as np
import tensorflow as tf

# 导入MNIST数据集
# from tensorflow.examples.tutorials.mnist import input_data
# mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
# batch_size = 100
#
# epoch_x, epoch_y = mnist.train.next_batch(batch_size)
# img = tf.placeholder('float', [batch_size, 28*28])
#
# ##
# def add_noise(img, batch_size):
#     noise_variables = tf.Variable(tf.random_normal([784]), name="noise_variables")
#     noise_variables = tf.clip_by_value(noise_variables, -0.2, 0.2)
#     img = tf.convert_to_tensor(img)
#     print('img=', img)
#     img_2 = tf.Variable(tf.zeros([100, 784], dtype=float))
#     # img_out = tf.Variable(tf.ones([100, 784], dtype=float))
#     print('img_2=', img_2)
#     print('noise_variables=', noise_variables)
#     for i in range(batch_size):
#         img_single = tf.add(img[i, :], noise_variables)
#         img_2 = tf.scatter_nd()
#         print('epoch:', i)
#     img_out = tf.clip_by_value(img_out, 0, 1)
#
#     return img_out
#
# ##
#
#
# images_with_noise = add_noise(img, batch_size)
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     images_out = sess.run(images_with_noise, feed_dict={img: epoch_x})
# ##
#     images_out1 = np.reshape(images_out[98, :], (28, 28))
#     plt.imshow(images_out1)
#
# ##
# a = tf.Variable(tf.zeros([3, 3], dtype=float))
# b = tf.Variable(tf.ones([3], dtype=float))
# print('a=', a)
# print('b=', b)
# c = tf.add(a[0, :], b)
# print('c=', c)
#
# d = tf.scatter_nd_update(a, [1, 0], c)
#
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     a2 = sess.run(d)


# def gen():
#     with tf.variable_scope('test'):
#         var1 = tf.get_variable('a1', shape=[1], dtype=float)
#         print(var1)
#         b = tf.Variable(1.0)
#         var2 = tf.assign(var1[0], b)
#         return var2
#
#
# a3 = gen()
# with tf.variable_scope('test', reuse=True):
#     var2 = tf.get_variable('a1')
#     with tf.Session() as sess:
#         sess.run(tf.global_variables_initializer())
#         print(sess.run([a3, var2]))
##
a = tf.Variable(tf.zeros([100, 784], dtype=float))
b = a.get_shape()[0]
print(b)
