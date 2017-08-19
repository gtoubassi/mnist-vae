import tensorflow as tf
import numpy as np
import png
from tensorflow.examples.tutorials.mnist import input_data
from dataset import Dataset
from trainer import Trainer
from mnist_fc import MNISTFullyConnected
import pdb
import argparse
import os

# XXX try leaky vs non leaky
# XXX try mean squared vs cross entropy
# XXX batch norm
# batch size of 100 per paper
# predict z_log_sigma instead of sigma
#stddev .02 per kvfrans

class MNISTConvVariationalAutoencoder:

  def __init__(self, vae, embedding_size, batchsize):
    num_hidden = 256
    self.batchsize = batchsize

    with tf.variable_scope("autoencoder"):
      # input layer
      with tf.variable_scope("input"):
        self.x = tf.placeholder(tf.float32, shape=[None, 784])
        x_image = tf.reshape(self.x, [-1,28,28,1])

      # First layer (conv)
      with tf.variable_scope("conv1"):
        W_conv1 = self.weight_variable([5, 5, 1, 32])
        b_conv1 = self.bias_variable([32])

        h_conv1 = self.relu(self.conv2d(x_image, W_conv1) + b_conv1)
        h_pool1 = self.max_pool_2x2(h_conv1)

      # Second layer (conv)
      with tf.variable_scope("conv2"):
        W_conv2 = self.weight_variable([5, 5, 32, 64])
        b_conv2 = self.bias_variable([64])

        h_conv2 = self.relu(self.conv2d(h_pool1, W_conv2) + b_conv2)
        h_pool2 = self.max_pool_2x2(h_conv2)

      # Third layer (fully connected)
      with tf.variable_scope("fc1"):
        W_fc1 = self.weight_variable([7 * 7 * 64, 1024])
        b_fc1 = self.bias_variable([1024])

        h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
        h_fc1 = self.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

      # Fourth layer (embedding layer)
      with tf.variable_scope("embed-fc2"):
        if vae:
          W_mean = self.weight_variable([1024, embedding_size], 'mean-weights')
          b_mean = self.bias_variable([embedding_size], 'mean-biases')
          z_mean = tf.matmul(h_fc1, W_mean) + b_mean
          self.embed_layer = z_mean

          W_stddev = self.weight_variable([1024, embedding_size], 'stddev-weights')
          b_stddev = self.bias_variable([embedding_size], 'stddev-biases')
          z_stddev = tf.matmul(h_fc1, W_stddev) + b_stddev

          samples = tf.random_normal([batchsize, embedding_size])
          self.z = z_mean + samples * z_stddev
        else:
          W_fc2 = self.weight_variable([1024, embedding_size])
          b_fc2 = self.bias_variable([embedding_size])
          h_fc2 = self.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)
          self.embed_layer = h_fc2
          self.z = h_fc2

      # Fifth layer (fully connected)
      with tf.variable_scope("fc3-decoder"):
        W_fc3 = self.weight_variable([embedding_size, 1024])
        b_fc3 = self.bias_variable([1024])

        h_fc3 = self.relu(tf.matmul(self.z, W_fc3) + b_fc3)

      # Sixth layer (fully connected)
      with tf.variable_scope("fc4-decoder"):
        W_fc4 = self.weight_variable([1024, 7*7*64])
        b_fc4 = self.bias_variable([7*7*64])

        h_fc4 = self.relu(tf.matmul(h_fc3, W_fc4) + b_fc4)
        h_fc4_reshaped = tf.reshape(h_fc4, [-1, 7, 7, 64])

      # Seventh layer (deconv)
      with tf.variable_scope("deconv1-decoder"):
        W_deconv1 = self.weight_variable([5, 5, 32, 64])
        b_deconv1 = self.bias_variable([32])
        h_dc1 = tf.nn.conv2d_transpose(h_fc4_reshaped, W_deconv1, output_shape=[self.batchsize, 14, 14, 32], strides=[1, 2, 2, 1])
        h_deconv1 = self.relu(h_dc1 + b_deconv1)

      with tf.variable_scope("deconv2-decoder"):
        W_deconv2 = self.weight_variable([5, 5, 1, 32])
        b_deconv2 = self.bias_variable([1])
        h_dc2 = tf.nn.conv2d_transpose(h_deconv1, W_deconv2, output_shape=[self.batchsize, 28, 28, 1], strides=[1, 2, 2, 1])
        self.y_logits = tf.reshape(h_dc2 + b_deconv2, [-1, 784])
        self.y = tf.nn.sigmoid(self.y_logits)

      self.y_ = tf.placeholder(tf.float32, [None, 784])

      generation_loss = -tf.reduce_sum(self.y_ * tf.log(1e-8 + self.y) + (1 - self.y_) * tf.log(1e-8 + 1 - self.y), 1)

      if vae:
        latent_loss = 0.5 * tf.reduce_sum(tf.square(z_mean) + tf.square(z_stddev) - tf.log(1e-8 + tf.square(z_stddev)) - 1,1)
        self.loss = tf.reduce_mean(generation_loss + latent_loss)
      else:
        self.loss = tf.reduce_mean(generation_loss)

      self.train_step = tf.train.AdamOptimizer().minimize(self.loss)
      self.accuracy = -self.loss
      self.session = tf.Session()
      self.session.run(tf.global_variables_initializer())
      writer = tf.summary.FileWriter("logs", self.session.graph)

  def relu(self, x, leak=0.2, name="myrelu"):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)

  def weight_variable(self, shape, name='weights'):
    return tf.get_variable(name, shape, initializer=tf.contrib.layers.xavier_initializer())

  def bias_variable(self, shape, name='biases'):
    return tf.get_variable(name, shape, initializer=tf.constant_initializer(0.0))

  def conv2d(self, x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

  def max_pool_2x2(self, x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

  def train_batch(self, x, y):
    self.session.run(self.train_step, feed_dict={self.x: x, self.y_: x})

  def evaluate(self, x, y):
    return self.session.run(self.accuracy, feed_dict={self.x: x, self.y_: x});

  def save(self, path):
    if not os.path.exists(path):
      os.makedirs(path)
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
    orig_image = np.reshape(x, (x.shape[0]*28, 28)) * 255.0
    result_image = np.reshape(result, (x.shape[0]*28, 28)) * 255.0
    combo_image = np.concatenate((orig_image, result_image), axis=1)
    png.save_png(output_path, combo_image)

  def generate_2d_digit_map(self, output_path):
    combo_image = None
    for x in range(20):
      z = np.zeros((100, 2))
      for y in range(20):
        z[y,0] = (x-10)/5.0
        z[y,1] = (y-10)/5.0
      result = self.session.run(self.y, feed_dict={self.z: z})
      result_image = np.reshape(result, (result.shape[0]*28, 28)) * 255.0
      result_image = result_image[:420,]
      if combo_image is None:
        combo_image = result_image
      else:
        combo_image = np.concatenate((combo_image, result_image), axis=1)
    png.save_png(output_path, combo_image)

def save_cluster_csv(z, y):
  y = [np.where(r==1)[0][0] for r in y]
  f = open('file.csv', 'w')
  for i in range(z.shape[0]):
    f.write("%f,%s%f\n" % (z[i,0], ',' * int(y[i]) ,z[i,1]))
  f.close

def main():

  parser = argparse.ArgumentParser()
  parser.add_argument("--vae", action='store_true', help="Use a vae vs a plain autoencoder")
  parser.add_argument("--save", help="Path to save the model to")
  parser.add_argument("--load", help="Path to load the model from (if unspecified will train)")
  parser.add_argument("--embedding-size", type=int, default=20, help="Dimensionality of the embedding (default 20)")
  parser.add_argument("--gen-digit-map", help="path to write a digit map")
  global args
  args = parser.parse_args()

  print(args)

  batchsize = 100
  mnist = Dataset.load_mnist()
  trainer = Trainer()

  autoencoder = MNISTConvVariationalAutoencoder(args.vae, args.embedding_size, batchsize)
  if args.load:
    print("Loading model from %s" % args.load)
    autoencoder.load(args.load)
  else:
    print("Training model...")
    accuracy = trainer.train(autoencoder, mnist, batchsize)
    print("Final for Accuracy %f" % accuracy)
    if args.save:
      autoencoder.save(args.save)

  if args.gen_digit_map:
    autoencoder.generate_2d_digit_map(args.gen_digit_map)
    exit()
    
  #autoencoder.output_reconstructed_images('out.png', mnist.test.batch(0,batchsize)[0])

  print("Generating embedding...")
  smallMnist = mnist.slice(2000, 1000, mnist.test.count)
  embeddedMnist = Dataset(autoencoder.embed(smallMnist.train.x, batchsize), mnist.train.y,
                          autoencoder.embed(smallMnist.validation.x, batchsize), mnist.validation.y,
                          autoencoder.embed(smallMnist.test.x, batchsize), mnist.test.y)

  print("Training on embedding...")
  classifier = MNISTFullyConnected(input_size=args.embedding_size)
  accuracy = trainer.train(classifier, embeddedMnist)
  print("Final for Embedded Accuracy %f" % accuracy)

if __name__ == "__main__":
    main()

