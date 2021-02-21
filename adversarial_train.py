#Train a noise pattern to attack the CNN
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os
# import numpy as np
# import matplotlib.pyplot as plt

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# one-hot code for labels
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
# parameters
n_classes = 10  # number of classifier 's output
batch_size = 100  # batch size
train_save_path = "./train_results/model"
# define placeholder
x = tf.placeholder('float', [None, 28 * 28])
y = tf.placeholder('float')
img = tf.placeholder('float', [batch_size, 28*28])
img_test = tf.placeholder('float', [10000, 28*28])
# define the convolution layer
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


# define  the pooling layer
def maxpool2d(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


# define the network configuration
def neural_network_model(data):
    # image size 28*28 -- 14*14 -- 7*7
    # using name scope to define variable
    with tf.name_scope(name='variable_for_disc'):

        # convolution 1
        w_conv1 = tf.Variable(tf.random_normal(shape=[5, 5, 1, 32]), name="w_conv1")
        # convolution 2
        w_conv2 = tf.Variable(tf.random_normal(shape=[5, 5, 32, 64]), name="w_conv2")
        # full connection
        w_fc = tf.Variable(tf.random_normal(shape=[7 * 7 * 64, 1024]), name="w_fc")
        # output layer
        w_out = tf.Variable(tf.random_normal(shape=[1024, n_classes]), name="w_out")

        #
        b_conv1 = tf.Variable(tf.random_normal([32]), name="b_conv1")
        #
        b_conv2 = tf.Variable(tf.random_normal([64]), name="b_conv2")
        #
        b_fc = tf.Variable(tf.random_normal([1024]), name="b_fc")
        #
        b_out = tf.Variable(tf.random_normal([n_classes]), name="b_out")

        # reshape the tensor
        data = tf.reshape(data, [-1, 28, 28, 1])

        #
        # conv + pooling
        conv1 = tf.nn.relu(conv2d(data, w_conv1) + b_conv1)
        conv1 = maxpool2d(conv1)

        # conv + pooling
        conv2 = tf.nn.relu(conv2d(conv1, w_conv2) + b_conv2)
        conv2 = maxpool2d(conv2)

        # full connection
        fc = tf.reshape(conv2, [-1, 7 * 7 * 64])
        fc = tf.nn.relu(tf.matmul(fc, w_fc) + b_fc)

        # output layer
        output = tf.matmul(fc, w_out) + b_out

        return output


# add noise on images
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
        intensity = tf.reduce_mean(tf.abs(noise_variables))

        return img_out, intensity  # intensity :average noise


img_with_noise, intensity = add_noise(img)
img_with_noise_larger, intensity2 = add_noise(img_test)
prediction = neural_network_model(x)
prediction_with_noise = neural_network_model(img_with_noise)
test_for_prediction_with_noise = neural_network_model(img_with_noise_larger)

loss0 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
loss1 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction_with_noise, labels=y))


correct_rate = tf.equal(tf.argmax(prediction_with_noise, 1), tf.argmax(y, 1))
accuracy_for_small_batch_with_noise = tf.reduce_mean(tf.cast(correct_rate, 'float'))
value_of_jam = accuracy_for_small_batch_with_noise + intensity

vars0 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='variable_for_disc')
vars1 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='variable_for_gen')
optimizer0 = tf.train.AdamOptimizer().minimize(loss0, var_list=vars0)
optimizer1 = tf.train.AdamOptimizer().minimize(value_of_jam, var_list=vars1)
optimizer2 = tf.train.AdamOptimizer().minimize(loss1, var_list=vars0)

hm_epochs = 10
saver = tf.train.Saver(var_list=vars0 + vars1, max_to_keep=1)
with tf.Session() as sess:
    # initialize
    sess.run(tf.global_variables_initializer())
    for epoch in range(hm_epochs):
        epoch_loss = 0
        v = 0
        for _ in range(int(mnist.train.num_examples / batch_size)):
            # load data
            epoch_x, epoch_y = mnist.train.next_batch(batch_size)
            # feed data
            _, c, v_j, noise_intensity = sess.run([optimizer1, loss1, value_of_jam, intensity],
                                                  feed_dict={x: epoch_x, y: epoch_y, img: epoch_x})
            epoch_loss = epoch_loss + c
            v = v + v_j - noise_intensity

        print('Epoch', epoch, 'completed out of', hm_epochs, 'loss', epoch_loss)
        print('Epoch', epoch, 'completed out of', hm_epochs, 'accuracy', v/550)
    # calculate the accuracy for classifier
    correct0 = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
    correct1 = tf.equal(tf.argmax(test_for_prediction_with_noise, 1), tf.argmax(y, 1))
    accuracy0 = tf.reduce_mean(tf.cast(correct0, 'float'))
    accuracy1 = tf.reduce_mean(tf.cast(correct1, 'float'))
    print('Accuracy0:', accuracy0.eval({x: mnist.test.images, y: mnist.test.labels}))
    print('Accuracy1:', accuracy1.eval({img_test: mnist.test.images, y: mnist.test.labels}))

    print("saving...")
    saver.save(sess, save_path=train_save_path)


print('hello world')





