__author__ = 'Aditya Gupta'
#"image_data_format": "channels_last"

#import shapes_3d
#import shapes_3d_M
import data_shapes_3d_cnn
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution3D, MaxPooling3D
from keras.optimizers import SGD, RMSprop
from keras.utils import np_utils, generic_utils
from keras import backend as K
from keras.models import Sequential
from keras.layers import Convolution3D, MaxPooling3D
from keras.layers.core import Activation, Dense, Dropout, Flatten
from keras.layers.advanced_activations import LeakyReLU
from keras.regularizers import l2
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from keras.optimizers import SGD
import theano
from numpy import matrix
import numpy as np
from keras import backend as K
from keras.regularizers import l2
K.set_image_dim_ordering('th')
import logging
import shapenet10

#to have consistent resutls
np.random.seed(1337)

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s| %(message)s')
logging.info('Loading Data...')

(X_train, Y_train),(X_test, Y_test),(patch_size) = data_shapes_3d_cnn.load_data()

logging.info('Data Specs::-')
logging.info(str('X_Train :: shape : ' + str(X_train.shape) + ' || type : ' + str(type(X_train))))
logging.info(str('Y_Train :: shape : ' + str(Y_train.shape) + '               || type : ' + str(type(Y_test))))
logging.info(str('X_Test :: shape  : ' + str(X_test.shape) + ' || type : ' + str(type(X_train))))
logging.info(str('Y_Test :: shape  : ' + str(Y_test.shape) + '               || type : ' + str(type(Y_test))))

# CNN Training parameters
batch_size = 15
nb_classes = shapenet10.nb_classes_current
nb_epoch = 5


#print(Y_test[0:10])
# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(Y_train, nb_classes)
Y_test = np_utils.to_categorical(Y_test, nb_classes)
#print(Y_test[0:10])
# number of convolutional filters to use at each layer
nb_filters = [16, 32]

# level of pooling to perform at each layer (POOL x POOL)
nb_pool = [3, 3]

# level of convolution to perform at each layer (CONV x CONV)
nb_conv = [7, 3]

#For now, the CNN is somewhat tweak version of Voxnet
#        layers:
#            3D Convolution
#            Leaky ReLu
#            Dropout
#            3d Convolution
#            Leaky ReLu
#            MaxPool
#            Dropout
#            Dense
#            Dropout
#            Dense
logging.info('Loading Model::-')

model = Sequential()

#Layer 1 : Convolution3D
model.add(Convolution3D(input_shape=(1, patch_size, patch_size, patch_size),
				nb_filter=32,
				kernel_dim1=5,
				kernel_dim2=5,
				kernel_dim3=5,
				init='normal',
				border_mode='valid',
				subsample=(2,2,2),
				dim_ordering='th',
				W_regularizer=l2(0.001),
                        b_regularizer=l2(0.001),
                        ))
logging.info("Layer1:Conv3D ")
#Layer 2 : Activation Leaky ReLu
model.add(Activation(LeakyReLU(alpha=0.1)))
logging.info("Layer2:LeakyRelu ")
#Layer 3 : Dropout 1
model.add(Dropout(p=0.3))
logging.info("Layer3:Dropout ")

model.add(Convolution3D(nb_filter=32,
                        kernel_dim1=3,
                        kernel_dim2=3,
                        kernel_dim3=3,
                        init='normal',
                        border_mode='valid',
                        subsample=(1, 1, 1),
                        dim_ordering='th',
                        W_regularizer=l2(0.001),
                        b_regularizer=l2(0.001),
                        ))
logging.info("Layer4:Conv3D ")

#Layer5 : Activation Leaky ReLu
model.add(Activation(LeakyReLU(alpha=0.1)))
logging.info("Layer%:Leaky Relu ")

#Layer 6 : max pool 1
model.add(MaxPooling3D(pool_size=(2, 2, 2),
                       strides=None,
                       border_mode='valid',
                       dim_ordering='th'))
logging.info("Layer6:Max Pool ")

#Layer 7 : Dropout
model.add(Dropout(0.4))
logging.info("Layer7: Dropout ")

#Layer 8 : Dense
model.add(Flatten())
model.add(Dense(output_dim=128,
                init='normal',
                activation='linear',
                W_regularizer=l2(0.001),
                b_regularizer=l2(0.001),
                ))
logging.info("Layer8:Dense ")

#Layer 9 : Dropout
model.add(Dropout(0.5))
logging.info("Layer9:Dropout ")

#Laye 10 : Fully conected Layer
model.add(Dense(output_dim=nb_classes,
                init='normal',
                activation='linear',
                W_regularizer=l2(0.001),
                b_regularizer=l2(0.001),
                ))
logging.info("Layer10 : Fully Connected Layer ")


model.add(Activation('softmax'))
logging.info("Softmax Classifier")

sgd = SGD(lr=0.01, momentum=0.9, decay=0.00016667, nesterov=False)

model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=["accuracy"])
logging.info("Model Compiled")

logging.info("Training...you can go take sleep :'D")
model.fit(X_train,
          Y_train,
	    batch_size=batch_size,
          epochs=nb_epoch,
          verbose=2,
          validation_data=(X_test, Y_test))

score = model.evaluate(X_test, Y_test, batch_size=batch_size, verbose=1)
#print("Basesline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
logging.info(str(score))

# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")

