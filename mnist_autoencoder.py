import tensorflow as tf
import numpy as np
import png
from tensorflow.examples.tutorials.mnist import input_data
from dataset import Dataset
from trainer import Trainer
from mnist_fc import MNISTFullyConnected
import pdb

class MNISTAutoencoder:
  
  def __init__(self, embedding_size):
    num_hidden = 256
    with tf.variable_scope("autoencoder"):
      self.x = tf.placeholder(tf.float32, [None, 784])

      # XXX should we try tf.random_normal_initializer(stddev=0.02)
      W1 = tf.get_variable("W1", shape=[784, num_hidden], initializer=tf.contrib.layers.xavier_initializer())
      b1 = tf.Variable(tf.constant(0.1, shape=[num_hidden]))
      h1 = tf.nn.relu(tf.matmul(self.x, W1) + b1)

      W2 = tf.get_variable("W2", shape=[num_hidden, embedding_size], initializer=tf.contrib.layers.xavier_initializer())
      b2 = tf.Variable(tf.constant(0.0, shape=[embedding_size]))
      h2 = tf.matmul(h1, W2) + b2
      self.embed_layer = h2

      W3 = tf.get_variable("W3", shape=[embedding_size, num_hidden], initializer=tf.contrib.layers.xavier_initializer())
      b3 = tf.Variable(tf.constant(0.1, shape=[num_hidden]))
      h3 = tf.nn.relu(tf.matmul(h2, W3) + b3)

      W4 = tf.get_variable("W4", shape=[num_hidden, 784], initializer=tf.contrib.layers.xavier_initializer())
      b4 = tf.Variable(tf.constant(0.0, shape=[784]))
      h4 = tf.nn.sigmoid(tf.matmul(h3, W4) + b4)

      self.y = h4

      self.y_ = tf.placeholder(tf.float32, [None, 784])

      self.loss = tf.reduce_mean(tf.squared_difference(self.y, self.y_))
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

  def embed(self, images, batchsize=100):
    num_batches = images.shape[0] // batchsize
    all_embeds = None
    for i in range(num_batches):
      batch = images[(i*batchsize):((i+1)*batchsize)]
      batch_embed = self.session.run(self.embed_layer, feed_dict={self.x: batch})
      if all_embeds is None:
        all_embeds = batch_embed
      else:
        all_embeds = np.concatenate((all_embeds, batch_embed))
    return all_embeds
  
  def output_reconstructed_images(self, output_path, x):
    result = self.session.run(self.y, feed_dict={self.x: x})
    orig_image = np.reshape(x, (100*28, 28)) * 255.0
    result_image = np.reshape(result, (100*28, 28)) * 255.0
    combo_image = np.concatenate((orig_image, result_image), axis=1)
    png.save_png(output_path, combo_image)

def main():

  embedding_size = 20
  mnist = Dataset.load_mnist()
  
  autoencoder = MNISTAutoencoder(embedding_size)
  trainer = Trainer()
  accuracy = trainer.train(autoencoder, mnist)
  print("Final for Accuracy %f" % accuracy)
  
  autoencoder.output_reconstructed_images('autoencoder.png', mnist.test.batch(0,100)[0])

  embeddedMnist = Dataset(autoencoder.embed(mnist.train.x), mnist.train.y,
                          autoencoder.embed(mnist.validation.x), mnist.validation.y,
                          autoencoder.embed(mnist.test.x), mnist.test.y)
  skinnyEmbeddedMNist = embeddedMnist.slice(2000, 1000, embeddedMnist.test.count)
  
  classifier = MNISTFullyConnected(input_size=embedding_size)
  accuracy = trainer.train(classifier, skinnyEmbeddedMNist)
  print("Final for Embedded Accuracy %f" % accuracy)

if __name__ == "__main__":
    main()

