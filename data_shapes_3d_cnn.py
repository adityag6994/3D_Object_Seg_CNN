#Get the data from computer and convert it into numpy array



__author__ = 'Aditya Gupta'

import sys, os
sys.path.append('/home/aditya/DR/keras_3d/keras/examples/3D_Object_Seg_CNN')
import logging
import random
import numpy as np
import scipy.io
from path import Path
import argparse
from numpy import matrix
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import shapenet10
#from shapenet10 import class_name_to_index
from matplotlib import cm
import h5py 
from sklearn.preprocessing import LabelEncoder


logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s| %(message)s')
#base_dir = Path('/home/aditya/DR/keras_3d/keras/examples/volumetric_data_D').expand()
base_dir = Path('/home/aditya/DR/keras_3d/keras/examples/3D_Object_Seg_CNN/volumetric_data').expand()

records = {'train': [], 'test': []}
logging.info('Loading .mat files')

#TODO : change this to do this on its own, and not explicit call
file_address = '/home/aditya/DR/keras_3d/keras/examples/volumetric_data/bottle/30/test/bottle_000000033_1.mat'
data = scipy.io.loadmat(file_address)
pcl_shape = data['instance'].shape
patch_size_x = pcl_shape[0]
patch_size_y = pcl_shape[1]
patch_size_z = pcl_shape[2]
#print type(records)
#print patch_size_x
#print len(records['train'])
Y_train_labels = list()
Y_test_labels = list()
count_train = 0
count_test = 0

#just to test with small dataset, 
count_train_TEST = 0
count_test_TEST  = 0

for fname in sorted(base_dir.walkfiles('*.mat')):
    if fname.endswith('test_feature.mat') or fname.endswith('train_feature.mat'):
        continue
    elts = fname.splitall()
    instance_rot = Path(elts[-1]).stripext()
    instance = instance_rot[:instance_rot.rfind('_')]
    rot = int(instance_rot[instance_rot.rfind('_')+1:])
    split = elts[-2]
    classname = elts[-4].strip()
    	
    if classname not in shapenet10.class_names:
        continue
    if split == 'train':
		Y_train_labels.append(shapenet10.class_name_to_id.get(classname))		
    else:
		Y_test_labels.append(shapenet10.class_name_to_id.get(classname))		

    records[split].append((classname, instance, rot, fname))


count_train = 0
Y_train = np.asarray(Y_train_labels, dtype=np.float32)
X_train =  np.zeros((len(records['train']),
				1,
				patch_size_x,
				patch_size_y,
				patch_size_z))

for i in records:
	if(i == 'train'):
		for j in records[i]:
			X_train[count_train] = scipy.io.loadmat(j[3])['instance'].astype(np.float32)
			count_train = count_train + 1


count_test = 0
Y_test = np.asarray(Y_test_labels, dtype=np.float32)
X_test =  np.zeros((len(records['test']),
				1,
				patch_size_x,
				patch_size_y,
				patch_size_z))

for i in records:
	if(i == 'test'):
		for j in records[i]:
			X_test[count_test] = scipy.io.loadmat(j[3])['instance']
			count_test = count_test + 1

#print data 
#logging.info(str('X_Train :: shape : ' + str(X_train.shape) + ' || type : ' + str(type(X_train))))
#logging.info(str('Y_Train :: shape : ' + str(Y_train.shape) + '               || type : ' + str(type(Y_test))))
#logging.info(str('X_Test :: shape  : ' + str(X_test.shape) + ' || type : ' + str(type(X_train))))
#logging.info(str('Y_Test :: shape  : ' + str(Y_test.shape) + '               || type : ' + str(type(Y_test))))

def load_data():

    
    # Getting the training set
    y_train = Y_train
    x_train = X_train

    # Getting the testing set
    y_test = Y_test
    x_test = X_test
	
    patch_size = patch_size_x

    return (x_train, y_train),(x_test, y_test),(patch_size)


