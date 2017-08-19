import tensorflow as tf
import numpy as np
import png
from tensorflow.examples.tutorials.mnist import input_data
from dataset import Dataset
from trainer import Trainer
from mnist_fc import MNISTFullyConnected
import pdb

def lrelu(x, leak=0.2, name="lrelu"):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)

class MNISTVariationalAutoencoder:
  
  def __init__(self, embedding_size, batchsize=32):    
    num_hidden = 256
    self.batchsize=batchsize
    # try xavier
    # try removing lrelu
    # try removing .001 for adamoptimizer step size
    # try bach size of 32
    with tf.variable_scope("autoencoder"):
      self.x = tf.placeholder(tf.float32, [None, 784])

      W1 = tf.get_variable("W1", shape=[784, num_hidden], initializer=tf.random_normal_initializer(stddev=0.02))
      b1 = tf.Variable(tf.constant(0.0, shape=[num_hidden]))
      h1 = tf.nn.relu(tf.matmul(self.x, W1) + b1)

      W2_mean = tf.get_variable("W2_mean", shape=[num_hidden, embedding_size], initializer=tf.random_normal_initializer(stddev=0.02))
      b2_mean = tf.Variable(tf.constant(0.0, shape=[embedding_size]))
      z_mean = tf.matmul(h1, W2_mean) + b2_mean
      self.embed_layer = z_mean

      W2_stddev = tf.get_variable("W2_stddev", shape=[num_hidden, embedding_size], initializer=tf.random_normal_initializer(stddev=0.02))
      b2_stddev = tf.Variable(tf.constant(0.0, shape=[embedding_size]))
      z_stddev = tf.matmul(h1, W2_stddev) + b2_stddev
      
      # should we predict log stddev as suggested in http://wiseodd.github.io/techblog/2016/12/10/variational-autoencoder/
      #XXX do we need the reparameterization trick?
      samples = tf.random_normal([self.batchsize, embedding_size])
      z = z_mean + samples * z_stddev

      W3 = tf.get_variable("W3", shape=[embedding_size, num_hidden], initializer=tf.random_normal_initializer(stddev=0.02))
      b3 = tf.Variable(tf.constant(0.1, shape=[num_hidden]))
      # XXX Should this be lrelu?
      h3 = tf.nn.relu(tf.matmul(z, W3) + b3)

      W4 = tf.get_variable("W4", shape=[num_hidden, 784], initializer=tf.random_normal_initializer(stddev=0.02))
      b4 = tf.Variable(tf.constant(0.0, shape=[784]))
      h4 = tf.nn.sigmoid(tf.matmul(h3, W4) + b4)

      self.y = h4

      self.y_ = tf.placeholder(tf.float32, [None, 784])

      # XXX need to add 1e-8
      generation_loss = tf.reduce_sum(-self.y_ * tf.log(1e-8 + self.y) - (1 - self.y_) * tf.log(1e-8 + 1 - self.y), axis=1)
      
      kl_loss = 0.5 * tf.reduce_sum(tf.square(z_mean) + tf.square(z_stddev) - tf.log(tf.square(z_stddev)) - 1, axis=1)

      self.loss = tf.reduce_mean(generation_loss + kl_loss)
      self.train_step = tf.train.AdamOptimizer().minimize(self.loss)
      self.accuracy = -tf.reduce_sum(self.loss)
      self.session = tf.Session()    
      self.session.run(tf.global_variables_initializer())
    
  def train_batch(self, x, y):
    self.session.run(self.train_step, feed_dict={self.x: x, self.y_: x})
  
  def evaluate(self, x, y):
    return self.session.run(self.accuracy, feed_dict={self.x: x, self.y_: x});

  def save(self, path):
    saver = tf.train.Saver(max_to_keep=1)
    saver.save(self.session, path + '/model')
    
  def load(self, path):
    saver = tf.train.Saver()
    saver.restore(self.session, tf.train.latest_checkpoint(path))

  def embed(self, images):
    return self.session.run(self.embed_layer, feed_dict={self.x:images})
  
  def output_reconstructed_images(self, output_path, x):
    result = self.session.run(self.y, feed_dict={self.x: x})
    orig_image = np.reshape(x, (x.shape[0]*28, 28)) * 255.0
    result_image = np.reshape(result, (x.shape[0]*28, 28)) * 255.0
    combo_image = np.concatenate((orig_image, result_image), axis=1)
    png.save_png(output_path, combo_image)

def main():

  embedding_size = 20
  batchsize = 32
  mnist = Dataset.load_mnist()
  
  autoencoder = MNISTVariationalAutoencoder(embedding_size, batchsize)
  trainer = Trainer()
  accuracy = trainer.train(autoencoder, mnist, batchsize)
  print("Final for Accuracy %f" % accuracy)
  
  autoencoder.output_reconstructed_images('vae.png', mnist.test.batch(0,batchsize)[0])

  embeddedMnist = Dataset(autoencoder.embed(mnist.train.x), mnist.train.y,
                          autoencoder.embed(mnist.validation.x), mnist.validation.y,
                          autoencoder.embed(mnist.test.x), mnist.test.y)
  skinnyEmbeddedMNist = embeddedMnist.slice(2000, 1000, embeddedMnist.test.count)
  
  classifier = MNISTFullyConnected(input_size=embedding_size)
  accuracy = trainer.train(classifier, skinnyEmbeddedMNist)
  print("Final for Embedded Accuracy %f" % accuracy)

if __name__ == "__main__":
    main()

