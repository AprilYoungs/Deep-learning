import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.datasets import cifar10
import sys
import time
import os
import shutil

epoch = 2000
batch_size = 128

tensor_borad_path = "/tmp/cifar10"
saver_path = "./saver/cifar_saved_session"
log_file = "./train_log.txt"


# define text labels (source: https://www.cs.toronto.edu/~kriz/cifar.html)
cifar10_labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

(train_x, train_y), (test_x, test_y) = cifar10.load_data()

# show test_data

# fig = plt.figure(figsize=(15,5))
# for i in range(36):
#     ax = fig.add_subplot(3, 12, i+1, xticks=[], yticks=[])
#     ax.imshow(test_x[i])
#     index = test_y[i][0]
#     title = cifar10_labels[index]
#     ax.set_title(title)
# plt.show()

# imgs shape [32, 32, 3]


def get_batch(X,Y,batch_size):

    num_batch = len(X)//batch_size

    for i in range(0, num_batch):
        yield X[i*batch_size:(i+1)*batch_size], Y[i*batch_size:(i+1)*batch_size]

# for batch_x, batch_y in get_batch(test_x, test_y, 100):
#     print(batch_x.shape, batch_y.shape)

train_graph = tf.Graph()

with train_graph.as_default():
    input_img = tf.placeholder(shape=[None, 32, 32, 3], dtype=tf.float32, name="images")
    labels = tf.placeholder(shape=[None, 1], dtype=tf.int64, name="labels")

    one_hot_lab = tf.one_hot(labels, 10)

    # 32
    conv1 = tf.layers.conv2d(input_img, 30, 3)
    # 30
    pool1 = tf.layers.max_pooling2d(conv1, 2, 2)
    # 15

    conv2 = tf.layers.conv2d(pool1, 60, 3)
    # 13
    pool2 = tf.layers.max_pooling2d(conv2, 2, 2)
    # 6

    conv3 = tf.layers.conv2d(pool2, 60, 3)
    # 4
    pool3 = tf.layers.max_pooling2d(conv3, 4, 4)
    # 1

    pool3 = tf.squeeze(pool3, axis=[1])

    logits = tf.layers.dense(pool3, 10)

    prediction = tf.identity(tf.nn.softmax(logits), name="prediction")

    loss = tf.losses.softmax_cross_entropy(one_hot_lab, logits)

    optimizer = tf.train.AdamOptimizer().minimize(loss)


    try:
        # DELETE the previous graph
        shutil.rmtree(tensor_borad_path)
    except:
        pass
    tf.summary.FileWriter(tensor_borad_path, train_graph)

# train network

with tf.Session(graph=train_graph) as sess:
    sess.run(tf.global_variables_initializer())

    for e in range(epoch):
        start = time.time()
        for i, (batch_img,batch_lab) in enumerate(get_batch(train_x, train_y, batch_size)):
            cost, _ = sess.run([loss, optimizer], feed_dict={input_img:batch_img,
                                                             labels:batch_lab})

            print("\r Epoch:{}, batch:{}, loss:{:.3f}".format(e, i, cost))
            sys.stdout.flush()

        end = time.time()-start
        with open(log_file, "a") as file:
            file.write("Epoch:{}, loss:{:.3f}, time_spent:{:.2f}s\n".format(e, cost, end))

    saver = tf.train.Saver()
    saver.save(sess, saver_path)
