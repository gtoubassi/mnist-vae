#!/usr/bin/env python
#
# From the MNIST tutorial: https://www.tensorflow.org/tutorials/mnist/pros/

import tensorflow as tf
import numpy as np
from dataset import Dataset
from trainer import Trainer

class MNISTConv:
  
  def __init__(self):
    with tf.variable_scope("mnist-conv"):
      # input layer
      with tf.variable_scope("input"):
        self.x = tf.placeholder(tf.float32, shape=[None, 784])
        self.y_ = tf.placeholder(tf.float32, shape=[None, 10])
        x_image = tf.reshape(self.x, [-1,28,28,1])

      # First layer (conv)
      with tf.variable_scope("conv1"):
        W_conv1 = self.weight_variable([5, 5, 1, 32])
        b_conv1 = self.bias_variable([32])

        h_conv1 = tf.nn.relu(self.conv2d(x_image, W_conv1) + b_conv1)
        h_pool1 = self.max_pool_2x2(h_conv1)

      # Second layer (conv)
      with tf.variable_scope("conv2"):
        W_conv2 = self.weight_variable([5, 5, 32, 64])
        b_conv2 = self.bias_variable([64])

        h_conv2 = tf.nn.relu(self.conv2d(h_pool1, W_conv2) + b_conv2)
        h_pool2 = self.max_pool_2x2(h_conv2)

      # Third layer (fully connected)
      with tf.variable_scope("fc1"):
        W_fc1 = self.weight_variable([7 * 7 * 64, 1024])
        b_fc1 = self.bias_variable([1024])

        h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

      # Fourth layer (final output)
      with tf.variable_scope("fc2"):
        W_fc2 = self.weight_variable([1024, 10])
        b_fc2 = self.bias_variable([10])

      y_logit = tf.matmul(h_fc1, W_fc2) + b_fc2

      self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_logit, labels=self.y_))
      self.train_step = tf.train.AdamOptimizer().minimize(self.loss)
      correct_prediction = tf.equal(tf.argmax(y_logit, 1), tf.argmax(self.y_, 1))
      self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
      self.session = tf.Session()
      self.session.run(tf.global_variables_initializer())

  def weight_variable(self, shape):
    return tf.get_variable('weights', shape, initializer=tf.contrib.layers.xavier_initializer())

  def bias_variable(self, shape):
    return tf.get_variable('biases', shape, initializer=tf.constant_initializer(0.0))

  def conv2d(self, x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

  def max_pool_2x2(self, x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

  def train_batch(self, x, y):
    self.session.run(self.train_step, feed_dict={self.x: x, self.y_: y})
  
  def evaluate(self, x, y):
    return self.session.run(self.accuracy, feed_dict={self.x: x, self.y_: y});
  
  def save(self, path):
    saver = tf.train.Saver(max_to_keep=1)
    saver.save(self.session, path + '/model')
    
  def load(self, path):
    saver = tf.train.Saver()
    saver.restore(self.session, tf.train.latest_checkpoint(path))

def main():

  mnist = Dataset.load_mnist()
  network = MNISTConv()
  trainer = Trainer()
  accuracy = trainer.train(network, mnist)
  print("Final Accuracy %f" % accuracy)

if __name__ == "__main__":
    main()
