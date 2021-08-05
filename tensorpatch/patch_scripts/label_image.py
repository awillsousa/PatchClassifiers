# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

import numpy as np
import tensorflow as tf
import os
import fnmatch
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
 
def load_graph(model_file):
  graph = tf.Graph()
  graph_def = tf.GraphDef()

  with open(model_file, "rb") as f:
    graph_def.ParseFromString(f.read())
  with graph.as_default():
    tf.import_graph_def(graph_def)

  return graph

def read_tensor_from_image_file(file_name, input_height=299, input_width=299,
				input_mean=0, input_std=255):
  input_name = "file_reader"
  output_name = "normalized"
  file_reader = tf.read_file(file_name, input_name)
  if file_name.endswith(".png"):
    image_reader = tf.image.decode_png(file_reader, channels = 3,
                                       name='png_reader')
  elif file_name.endswith(".gif"):
    image_reader = tf.squeeze(tf.image.decode_gif(file_reader,
                                                  name='gif_reader'))
  elif file_name.endswith(".bmp"):
    image_reader = tf.image.decode_bmp(file_reader, name='bmp_reader')
  else:
    image_reader = tf.image.decode_jpeg(file_reader, channels = 3,
                                        name='jpeg_reader')
  float_caster = tf.cast(image_reader, tf.float32)
  dims_expander = tf.expand_dims(float_caster, 0);
  resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
  normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
  sess = tf.Session()
  result = sess.run(normalized)

  return result

def load_labels(label_file):
  label = []
  proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
  for l in proto_as_ascii_lines:
    label.append(l.rstrip())
  return label

if __name__ == "__main__":
  file_name = "tf_files/flower_photos/daisy/3475870145_685a19116d.jpg"
  model_file = "tf_files/retrained_graph.pb"
  label_file = "tf_files/retrained_labels.txt"
  #input_height = 224
  #input_width = 224
  #input_mean = 128
  #input_std = 128
  #input_layer = "input"
  input_mean = 0
  input_std = 255
  input_height = 299
  input_width = 299
  input_layer = "Mul"
  output_layer = "final_result"

  parser = argparse.ArgumentParser()
  parser.add_argument("--dir_images", help="directory of image to be processed")
  parser.add_argument("--image", help="image to be processed")
  parser.add_argument("--graph", help="graph/model to be executed")
  parser.add_argument("--labels", help="name of file containing labels")
  parser.add_argument("--input_height", type=int, help="input height")
  parser.add_argument("--input_width", type=int, help="input width")
  parser.add_argument("--input_mean", type=int, help="input mean")
  parser.add_argument("--input_std", type=int, help="input std")
  parser.add_argument("--input_layer", help="name of input layer")
  parser.add_argument("--output_layer", help="name of output layer")
  args = parser.parse_args()

  if args.graph:
    model_file = args.graph
  if args.image:
    file_name = args.image
  if args.labels:
    label_file = args.labels
  if args.input_height:
    input_height = args.input_height
  if args.input_width:
    input_width = args.input_width
  if args.input_mean:
    input_mean = args.input_mean
  if args.input_std:
    input_std = args.input_std
  if args.input_layer:
    input_layer = args.input_layer
  if args.output_layer:
    output_layer = args.output_layer
  if args.dir_images:
    dir_images = args.dir_images

  graph = load_graph(model_file)
  if args.image:
     file_names = [file_name]
  else:
     file_names = [os.path.join(dir_images,f) for f in os.listdir(dir_images) if fnmatch.fnmatch(f,'*.jpg')]

  #t = read_tensor_from_image_file(file_name,
  #                                input_height=input_height,
  #                                input_width=input_width,
  #                                input_mean=input_mean,
  #                                input_std=input_std)

  input_name = "import/" + input_layer
  output_name = "import/" + output_layer
  input_operation = graph.get_operation_by_name(input_name);
  output_operation = graph.get_operation_by_name(output_name);
  result_clf = []
  labels = load_labels(label_file)
  with tf.Session(config = tf.ConfigProto(device_count = {'GPU': 2}),graph=graph) as sess:
      for n_file, imagem in enumerate(file_names):
          t = read_tensor_from_image_file(imagem,
                                  input_height=input_height,
                                  input_width=input_width,
                                  input_mean=input_mean,
                                  input_std=input_std)

          results = sess.run(output_operation.outputs[0],
                      {input_operation.outputs[0]: t})
          results = np.squeeze(results)

          top_k = results.argsort()[-5:][::-1]
          #labels = load_labels(label_file)
          print("Imagem: {0}".format(imagem))
          result_img = []
          for i in top_k:
             print(labels[i], results[i])

          label_pred = labels[np.argmax(results)]
          proba_pred = max(results) 
          label_real = (imagem.split('/')[-1]).split('_')[1]
          if label_real == "M": label_real = "malignos"
          else: label_real = "benignos"
          result_clf.append({'id': n_file, 'imagem': imagem, 'pred': label_pred, 'real': label_real, 'prob_pred': proba_pred})
  y_pred = []
  y_true = []
  y_scores = [] 
  for r in result_clf:
      y_pred.append(r['pred'])
      y_true.append(r['real'])
      y_scores.append(r['prob_pred'])

  cf_matrix = confusion_matrix(y_true, y_pred, labels=labels)      
  tn, fp, fn, tp = cf_matrix.ravel()
  print("Matriz de Confusao:")
  print(str(cf_matrix)) 
  print(classification_report(y_true, y_pred, target_names=labels))
  print("AUC (ROC): {}".format(roc_auc_score([0 if y == 'benignos' else 1 for y in y_true], y_scores)))
