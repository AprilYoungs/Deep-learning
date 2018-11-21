import numpy as np
import tensorflow as tf
from keras.datasets import cifar10


saver_path = "./saver/cifar_saved_session"

cifar10_labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

(train_x, train_y), (test_x, test_y) = cifar10.load_data()

# test network

load_graph = tf.Graph()
with tf.Session(graph=load_graph) as sess:
    loader = tf.train.import_meta_graph(saver_path + ".meta")
    loader.restore(sess, saver_path)

    imgs = load_graph.get_tensor_by_name("images:0")
    predict = load_graph.get_tensor_by_name("prediction:0")

    prediction_num = sess.run(predict, feed_dict={imgs: test_x})

    prediction_num = np.squeeze(prediction_num)
    test_y = np.squeeze(test_y)
    result = np.argmax(prediction_num, axis=1)
    accuracy = sum(test_y == result) / len(test_y)

    print("Test accuracy:{:.3f}%".format(accuracy * 100))