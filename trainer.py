import tensorflow as tf
import numpy as np
import time
import shutil
import tempfile

class Trainer:

  # Train with early stopping (as soon as we regress w.r.t. the validation dataset)
  def train(self, network, dataset, batchsize=32):
    path = tempfile.mkdtemp()
    
    best_accuracy = -1000000
    num_backward = 0
    for epoch in range(100000):
      dataset.train.shuffle()
      start = time.time()
      for i in range(int(dataset.train.count / batchsize)):
        network.train_batch(*dataset.train.batch(i, batchsize))
      accuracy = self.eval(network, dataset.validation, batchsize)
      duration = time.time() - start
      print("Epoch %d Accuracy %f (%.3fs)" % (epoch, accuracy, duration))
      if accuracy > best_accuracy:
        best_accuracy = accuracy
        network.save(path)
        num_backward = 0
      else:
        num_backward += 1
        if num_backward > 2:
          break
    
    network.load(path)
    shutil.rmtree(path)
    return self.eval(network, dataset.test, batchsize)

  def eval(self, network, examples, batchsize=32):
    accuracy = 0
    num_batches = examples.count // batchsize
    for i in range(num_batches):
      accuracy += network.evaluate(*examples.batch(i, batchsize))
    return accuracy / num_batches
    
