import tensorflow as tf
import numpy as np
from dataset import Dataset
from trainer import Trainer
from mnist_conv_vae import MNISTConvVariationalAutoencoder
import argparse

class MNISTWideDeep:
  
  def __init__(self):
    with tf.variable_scope("mnist-fc"):
      self.x = tf.placeholder(tf.float32, [None, 784+20])
      
      pixels = tf.slice(self.x, [0, 0], [32, 784], 'pixels')
      latent_code = tf.slice(self.x, [0, 784], [32, 20], 'latent_code')
      
      #print(pixels)
      #print(latent_code)

      W1 = tf.get_variable("W1", shape=[784, 100], initializer=tf.contrib.layers.xavier_initializer())
      b1 = tf.Variable(tf.constant(0.1, shape=[100]))
      h1 = tf.nn.relu(tf.matmul(pixels, W1) + b1)
      
      h1 = tf.concat([h1, latent_code], axis=1)
      
      W2 = tf.get_variable("W2", shape=[120, 10], initializer=tf.contrib.layers.xavier_initializer())
      b2 = tf.Variable(tf.constant(0.1, shape=[10]))
      h2 = tf.matmul(h1, W2) + b2

      y_logits = h2
      self.y_ = tf.placeholder(tf.float32, [None, 10])

      self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_logits, labels=self.y_))

      self.train_step = tf.train.AdamOptimizer().minimize(self.loss)

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
  parser.add_argument("--embed-model", help="path to the model to use for embedding")
  global args
  
  args = parser.parse_args()    
  
  print(args)

  batchsize = 100
  
  trainer = Trainer()

  mnist = Dataset.load_mnist()
  autoencoder = MNISTConvVariationalAutoencoder(True, 20, batchsize)
  print("Loading model from %s" % args.embed_model)
  autoencoder.load(args.embed_model)

  print("Generating embedding...")
  smallMnist = mnist.slice(2000, 1000, mnist.test.count)
  embeddedMnist = Dataset(autoencoder.embed(smallMnist.train.x, batchsize), mnist.train.y,
                          autoencoder.embed(smallMnist.validation.x, batchsize), mnist.validation.y,
                          autoencoder.embed(smallMnist.test.x, batchsize), mnist.test.y)
  embeddedMnist.train.x = np.concatenate((smallMnist.train.x, embeddedMnist.train.x), axis=1)
  embeddedMnist.validation.x = np.concatenate((smallMnist.validation.x, embeddedMnist.validation.x), axis=1)
  embeddedMnist.test.x = np.concatenate((smallMnist.test.x, embeddedMnist.test.x), axis=1)

  print("Training on embedding...")
  classifier = MNISTWideDeep()
  accuracy = trainer.train(classifier, embeddedMnist)
  print("Final for Embedded Accuracy %f" % accuracy)

if __name__ == "__main__":
    main()
