# Modules

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import sys
import os
from PIL import Image, ImageOps
import csv

#Variables d'environnement
STEP_NUMBER = os.environ.get('STEP_NUMBER', 200)
MODEL_PATH = os.environ.get('MODEL_PATH', './model')
PERCENTILE_LIMIT = int(os.environ.get('PERCENTILE_LIMIT', 90))
FILE_SIZE = 28,28

tf.logging.set_verbosity(tf.logging.INFO)

# Définition du modèles de réseau neuronal
def cnn_model_fn(features, labels, mode):
 
  input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])

  conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=32,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)

  pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

  conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=64,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)

  pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

  pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])

  dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)

  dropout = tf.layers.dropout(
      inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

  logits = tf.layers.dense(inputs=dropout, units=10)

  predictions = {
      "classes": tf.argmax(input=logits, axis=1),
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
  }
  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    train_op = optimizer.minimize(
        loss=loss,
        global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

  eval_metric_ops = {
      "accuracy": tf.metrics.accuracy(
          labels=labels, predictions=predictions["classes"])}
  return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

# Fonction principale
def main(argv):
  argv = argv[1:]
  #print(argv)
  sess = tf.Session()
  sess.run(tf.global_variables_initializer())

  # On crée l'estimator
  mnist_classifier = tf.estimator.Estimator(
      model_fn=cnn_model_fn, model_dir=MODEL_PATH)

  tensors_to_log = {"probabilities": "softmax_tensor"}
  logging_hook = tf.train.LoggingTensorHook(
      tensors=tensors_to_log, every_n_iter=50)
  
  if len(argv) == 0:
    # On charge les données d'entrainement
    
    # Entrainement CSV ( Deprecié )
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
    #Entrainement MNIST (actuel)
    train_data = mnist.train.images  # Returns np.array
    train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
    eval_data = mnist.test.images  # Returns np.array
    eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": eval_data[:25]},
      y=eval_labels[:25],
      num_epochs=1,
      shuffle=False)
  
    # Entraine le modèle
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

    # Evalue le modèle et affiche les résultats  
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
        im.thumbnail(FILE_SIZE, Image.ANTIALIAS) # Convertit l'image a la taille désirée
        im = im.convert('L') # la convertie en niveaux de gris
        im = ImageOps.invert(im)
        newImg = Image.new('L', FILE_SIZE)
        box = (
            int((FILE_SIZE[0] - im.size[0]) / 2),
            int((FILE_SIZE[1] - im.size[1]) / 2)
          )
        newImg.paste(im, box) # Réduit la taille de l'image
        #newImg.save("{}_compressed.jpg".format(os.path.splitext(infile)[0]))
        pix = newImg.load() # Get the pixels values
        
        #Réduction du bruit
        fileData = []
        for y in range(FILE_SIZE[0]):
          for x in range(FILE_SIZE[1]):
            fileData.append((pix[x,y] / 255)) 
        limit = np.percentile(fileData, PERCENTILE_LIMIT)
        fileData = [1 if v > limit else 0 for v in fileData] # We use a percentile limit to remove noise
        Image.fromarray(np.array([v*255 for v in fileData], dtype=np.uint8).reshape(28,28)).save("{}_compressed.jpg".format(os.path.splitext(infile)[0]))
        filesDataList.append(fileData)
        
    # Evaluation du modèle via l'image recue
    filesDataList = np.array(filesDataList, dtype=np.float32)
    filesDataList = np.array(filesDataList)
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": filesDataList},
      num_epochs=1,
      shuffle=False)
    
    # Récupère les résultats
    eval_results = list(mnist_classifier.predict(input_fn=eval_input_fn))
    eval_results = [p["classes"] for p in eval_results]
    decalage = 0
    
    # On affiche NULL si on avais NULL en paramètre, sinon on affiche le résultat
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