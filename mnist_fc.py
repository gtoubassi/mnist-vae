import tensorflow as tf
import numpy as np
from dataset import Dataset
from trainer import Trainer
import argparse

class MNISTFullyConnected:
  
  def __init__(self, input_size=784):
    with tf.variable_scope("mnist-fc"):
      self.x = tf.placeholder(tf.float32, [None, input_size])

      W1 = tf.get_variable("W1", shape=[input_size, 100], initializer=tf.contrib.layers.xavier_initializer())
      b1 = tf.Variable(tf.constant(0.1, shape=[100]))
      h1 = tf.nn.relu(tf.matmul(self.x, W1) + b1)

      W2 = tf.get_variable("W2", shape=[100, 10], initializer=tf.contrib.layers.xavier_initializer())
      b2 = tf.Variable(tf.constant(0.1, shape=[10]))
      h2 = tf.matmul(h1, W2) + b2

      y_logits = h2
      self.y_ = tf.placeholder(tf.float32, [None, 10])

      self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_logits, labels=self.y_))

      self.train_step = tf.train.AdamOptimizer().minimize(self.loss)
      #self.train_step = tf.train.GradientDescentOptimizer(.005).minimize(self.loss)

      correct_prediction = tf.equal(tf.argmax(y_logits,1), tf.argmax(self.y_,1))
      self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
      self.session = tf.Session()
      self.session.run(tf.global_variables_initializer())

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
  parser = argparse.ArgumentParser()
  parser.add_argument("--train-size", help="number of training samples to use (e.g. --train-size 1000).  By default use all")
  parser.add_argument("--validation-size", help="number of validation samples to use (e.g. --validation-size 500).  By default use all")
  parser.add_argument("--batch-size", type=int, default=32, help="training and eval batch size (default 32).")
  global args
  
  args = parser.parse_args()    
  
  print(args)

  mnist = Dataset.load_mnist()
  if args.train_size or args.validation_size:
    train_size = int(args.train_size) if args.train_size else mnist.train.count
    validation_size = int(args.validation_size) if args.validation_size else mnist.validation.count
    mnist = mnist.slice(train_size, validation_size, mnist.test.count)

  network = MNISTFullyConnected()
  trainer = Trainer()
  accuracy = trainer.train(network, mnist, batchsize=args.batch_size)
  print("Final for Accuracy %f" % accuracy)

if __name__ == "__main__":
    main()
