# import the necessary packages
from cnn import Inception 
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.externals import joblib
from sklearn.datasets import dump_svmlight_file
from sklearn import preprocessing
from imutils import paths
import os
import argparse
import re
import cv2
import datetime

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,
	help="path to input images")
ap.add_argument("-o", "--output", required=True,
	help="output filename")
ap.add_argument("-p", "--path", required=True,
	help="path from cnn model")
args = vars(ap.parse_args())
output = args["output"]
patchs = [[960,1280],
          [480,640],
          [240,320],
          [120,160]]

##patchs = [[120,160],
##          [96,128],
##          [48,64]]

length = len(patchs)

hashmap = dict()

layer="pool_3:0"
cnn_path = args["path"]
desc = Inception(cnn_path,layer)
for pi in range(0,length):
        # initialize the CNN descriptor along with
        # the data and label lists
        data = []
        labels = []
        numlabels = []
        numlabel = 0
        # crop dimensions
        h = 960
        w = 1280
        ph = patchs[pi][0]# patch height
        pw = patchs[pi][1]# patch width
        patchstring = str(ph)+'x'+str(pw)
        print('========================================================')
        print(patchstring)
        print('Start',(datetime.datetime.now()))
        # loop over the input images
        for imagePath in os.listdir(args["input"]):
                abspath = os.path.join(args["input"], imagePath);
                # extract the label from the image path, then update the
                # label and data lists
                clabel = os.path.split(abspath)[-1]
                spt = clabel.split("_")
                label = re.sub(spt[len(spt)-1], "", clabel)
                if label not in hashmap:
                        numlabel +=1
                        hashmap[label] = numlabel
                else:
                        numlabel = hashmap[label]
                #print numlabel
                # load the image, convert it to grayscale, and describe it
                gray = cv2.imread(abspath,0)
                # crop image before compute features
                crop = gray[0:h,0:w]
                ch,cw = crop.shape[:2]
                if ph==h and pw==w:
                    patch = crop[0:ph,0:pw]
                    # extract features using CNN
                    hist = desc.describe(patch)
                    labels.append(label)
                    numlabels.append(float(numlabel))
                    data.append(hist)
                else:
                    for i in range(0,ch,ph):
                            for j in range(0,cw,pw):
                                    patch = crop[i:i+ph,j:j+pw]
                                    # extract features using CNN
                                    hist = desc.describe(patch)
                                    labels.append(label)
                                    numlabels.append(float(numlabel))
                                    data.append(hist)
        dump_svmlight_file(data,numlabels, output+'_cnn_'+patchstring+'.dat',)
        print('End',(datetime.datetime.now()))


