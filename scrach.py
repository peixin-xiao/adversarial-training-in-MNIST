##
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# 导入MNIST数据集
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
batch_size = 100

epoch_x, epoch_y = mnist.train.next_batch(batch_size)
img = tf.placeholder('float', [batch_size, 28*28])
# images = mnist.test.images[0:24]

##
# Plot the images and labels using our helper-function above.
# for i in range(24):
#     image = images[i, :]
#     image = np.reshape(image, (28, 28), order='C')
#     plt.subplot(5, 5, i+1)
#     plt.imshow(image)


##
# image1 = images[0, :]
# image2 = images[1, :]
# image1 = np.reshape(image1, (28, 28), order='C')
# image2 = np.reshape(image2, (28, 28), order='C')
# image3 = image1+image2
# plt.imshow(image3)


##
# a = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]], dtype=int)
# a = a[0, :]
# print(a)
# for i in range(10):
#     print(i, '\n')
# print(abs(-1))
##
def add_noise(img, batch_size):
    noise_variables = tf.Variable(tf.random_normal([1, 784]), name="noise_variables")
    noise_variables1 = tf.clip_by_value(noise_variables, -0.2, 0.2)
    img = tf.convert_to_tensor(img)
    img_out = tf.add(img[0, :], noise_variables1)
    img_out = tf.reshape(img_out, (1, 784))
    for i in range(1, batch_size):
        img_single = (tf.add(img[i, :], noise_variables1))
        img_out = tf.concat([img_out, img_single], 0)
    img_out = tf.clip_by_value(img_out, 0, 1)
    # flag = tf.get_variable('a1', shape=[1], dtype=float)
    # intensity = tf.assign(flag[0], tf.reduce_mean(tf.abs(noise_variables1)))
    intensity = tf.reduce_mean(tf.abs(noise_variables1))

    return img_out, intensity

##


images_with_noise, intensity = add_noise(img, batch_size)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    images_out, intensity = sess.run([images_with_noise, intensity], feed_dict={img: epoch_x})
##
    images_out1 = np.reshape(images_out[0, :], (28, 28))
    plt.imshow(images_out1)


# image = images[0, :]
# image = np.reshape(image, (28, 28), order='C')
# noise2, intensity = add_noise(image)
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     noise3, intensity1 = sess.run([noise2, intensity])
#     plt.imshow(noise3)
#     print(intensity1)

##
# a = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], dtype=int)
# b = np.reshape(a, (3, 2, 2), order='C')
# print(b)





