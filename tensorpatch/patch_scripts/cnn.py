import os
from os.path import basename
import tensorflow as tf
import tensorflow.python.platform
from tensorflow.python.platform import gfile
import numpy as np
import cv2

class Inception:
        def __init__(self, cnn_path, layer):
                # store the cnn path and the layer to extract features
                self.path=cnn_path
                self.layer=layer
                # store objects from model to extract features
                self.flattened_tensor=None
                self.sess=None
                # load the model
                self.create_graph(self.path)
        def create_graph(self, model_path):
                """
                create_graph loads the inception model to memory, should be called before
                calling describe. The __init__ does that.
             
                model_path: path to inception model in protobuf form.
                """
                with gfile.FastGFile(model_path, 'rb') as f:
                        graph_def = tf.GraphDef()
                        graph_def.ParseFromString(f.read())
                        _ = tf.import_graph_def(graph_def, name='')
                with tf.Session() as sess:
                        flattened_tensor = sess.graph.get_tensor_by_name(self.layer)
                        self.flattened_tensor = flattened_tensor
                self.sess = sess
        def describe(self, image):
                """
                extract_features computed the inception bottleneck feature for an image
             
                image: image
                return: feature vector of 2048 components
                """
                feature_dimension = 2048
                # create an array with 2048 positions
                features = np.zeros((feature_dimension))
                # convert an grayscale image to rgb
                # the original inception model requires a color image 
                np_image = cv2.cvtColor(image,cv2.COLOR_GRAY2RGB)
                # extract features
                feature = self.sess.run(self.flattened_tensor, {'DecodeJpeg:0': np_image})
                features = np.squeeze(feature)
                return features

