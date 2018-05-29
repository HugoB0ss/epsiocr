from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import sys
import os
from PIL import Image, ImageOps
import csv

STEP_NUMBER = os.environ.get('STEP_NUMBER', 200)
MODEL_PATH = os.environ.get('MODEL_PATH', './model')
PERCENTILE_LIMIT = int(os.environ.get('PERCENTILE_LIMIT', 90))
FILE_SIZE = 28,28

tf.logging.set_verbosity(tf.logging.INFO)

def cnn_model_fn(features, labels, mode):
  """Model function for CNN."""
  # Input Layer
  # Reshape X to 4-D tensor: [batch_size, width, height, channels]
  # MNIST images are 28x28 pixels, and have one color channel
  input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])

  # Convolutional Layer #1
  # Computes 32 features using a 5x5 filter with ReLU activation.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 28, 28, 1]
  # Output Tensor Shape: [batch_size, 28, 28, 32]
  conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=32,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)

  # Pooling Layer #1
  # First max pooling layer with a 2x2 filter and stride of 2
  # Input Tensor Shape: [batch_size, 28, 28, 32]
  # Output Tensor Shape: [batch_size, 14, 14, 32]
  pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

  # Convolutional Layer #2
  # Computes 64 features using a 5x5 filter.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 14, 14, 32]
  # Output Tensor Shape: [batch_size, 14, 14, 64]
  conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=64,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)

  # Pooling Layer #2
  # Second max pooling layer with a 2x2 filter and stride of 2
  # Input Tensor Shape: [batch_size, 14, 14, 64]
  # Output Tensor Shape: [batch_size, 7, 7, 64]
  pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

  # Flatten tensor into a batch of vectors
  # Input Tensor Shape: [batch_size, 7, 7, 64]
  # Output Tensor Shape: [batch_size, 7 * 7 * 64]
  pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])

  # Dense Layer
  # Densely connected layer with 1024 neurons
  # Input Tensor Shape: [batch_size, 7 * 7 * 64]
  # Output Tensor Shape: [batch_size, 1024]
  dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)

  # Add dropout operation; 0.6 probability that element will be kept
  dropout = tf.layers.dropout(
      inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

  # Logits layer
  # Input Tensor Shape: [batch_size, 1024]
  # Output Tensor Shape: [batch_size, 10]
  logits = tf.layers.dense(inputs=dropout, units=10)

  predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      "classes": tf.argmax(input=logits, axis=1),
      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
      # `logging_hook`.
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
  }
  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  # Calculate Loss (for both TRAIN and EVAL modes)
  loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

  # Configure the Training Op (for TRAIN mode)
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    train_op = optimizer.minimize(
        loss=loss,
        global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

  # Add evaluation metrics (for EVAL mode)
  eval_metric_ops = {
      "accuracy": tf.metrics.accuracy(
          labels=labels, predictions=predictions["classes"])}
  return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def main(argv):
  argv = argv[1:]
  #print(argv)
  sess = tf.Session()
  sess.run(tf.global_variables_initializer())

  # Create the Estimator
  mnist_classifier = tf.estimator.Estimator(
      model_fn=cnn_model_fn, model_dir=MODEL_PATH)

  # Set up logging for predictions
  # Log the values in the "Softmax" tensor with label "probabilities"
  tensors_to_log = {"probabilities": "softmax_tensor"}
  logging_hook = tf.train.LoggingTensorHook(
      tensors=tensors_to_log, every_n_iter=50)
  
  if len(argv) == 0:
    # Load training and eval data
    
    #CSV LOAD
    """with open('train.csv', 'r') as trainFile:
      train_data = []
      train_labels = []
      reader = csv.reader(trainFile, delimiter=',')
      firstline = True
      for row in reader:
        if firstline:    #skip first line
          firstline = False
          continue
        train_labels.append(row.pop(0))
        row = np.array(row, dtype=np.float32)
        train_data.append(row)
      train_data = np.array(train_data)
      train_labels = np.array(train_labels, dtype=np.int32)"""

    mnist = tf.contrib.learn.datasets.load_dataset("mnist")
    #MNIST LOAD
    train_data = mnist.train.images  # Returns np.array
    train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
    eval_data = mnist.test.images  # Returns np.array
    eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": eval_data[:25]},
      y=eval_labels[:25],
      num_epochs=1,
      shuffle=False)
  
    # Train the model
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": train_data},
      y=train_labels,
      batch_size=100,
      num_epochs=None,
      shuffle=True)
    
    mnist_classifier.train(
      input_fn=train_input_fn,
      steps=STEP_NUMBER,
      hooks=[logging_hook])

    # Evaluate the model and print results  
    eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
    #print(eval_results)
  else:
    filesDataList = []
    for infile in argv:
      if infile == "NULL":
        continue
      outfile = os.path.splitext(infile)[0] + ".thumbnail"
      if infile != outfile:
        im = Image.open(infile)
        im.thumbnail(FILE_SIZE, Image.ANTIALIAS) # Convert to FILE_SIZE size
        im = im.convert('L')
        im = ImageOps.invert(im)
        newImg = Image.new('L', FILE_SIZE)
        box = (
            int((FILE_SIZE[0] - im.size[0]) / 2),
            int((FILE_SIZE[1] - im.size[1]) / 2)
          )
        newImg.paste(im, box)
        #newImg.save("{}_compressed.jpg".format(os.path.splitext(infile)[0]))
        pix = newImg.load() # Get the pixels values
        fileData = []
        for y in range(FILE_SIZE[0]):
          for x in range(FILE_SIZE[1]):
            fileData.append((pix[x,y] / 255))
        limit = np.percentile(fileData, PERCENTILE_LIMIT)
        fileData = [1 if v > limit else 0 for v in fileData] # We use a percentile limit to remove noise
        Image.fromarray(np.array([v*255 for v in fileData], dtype=np.uint8).reshape(28,28)).save("{}_compressed.jpg".format(os.path.splitext(infile)[0]))
        filesDataList.append(fileData)
        
    filesDataList = np.array(filesDataList, dtype=np.float32)
    filesDataList = np.array(filesDataList)
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": filesDataList},
      num_epochs=1,
      shuffle=False)
    
    eval_results = list(mnist_classifier.predict(input_fn=eval_input_fn))
    eval_results = [p["classes"] for p in eval_results]
    decalage = 0
    for i,r in enumerate(argv):
        if argv[i] == "NULL":
          decalage += 1
          print("-1")
        else:
            print(eval_results[i-decalage])
    """eval_results = ["{}|{}".format(argv[i], r) for i,r in enumerate(eval_results)]
    for result in eval_results:
        print(result)
    """
if __name__ == "__main__":
  #print(sys.argv)
  tf.app.run()
#https://opencv.org/