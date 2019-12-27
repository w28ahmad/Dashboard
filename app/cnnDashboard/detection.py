from darkflow.defaults import argHandler #Import the default arguments
import os
from darkflow.net.build import TFNet

# object detection parameters
FLAGS = {
         'imgdir': 'sample_img/', 
         'binary': './bin/', 
         'config': './cfg/', 
         'backup': './ckpt/', 
         'threshold': -0.1, 
         'model': 'cfg/yolo.cfg', 
         'load': 'bin/yolo.weights', 
         'gpu': 0.0, 
         'batch': 16
         }
tfnet = TFNet(FLAGS) 