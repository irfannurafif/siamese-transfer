#%load_ext autoreload
#%autoreload 2
#import autosklearn.classification
import os

import sklearn.model_selection
import sklearn.datasets
import sklearn.metrics
from keras.datasets import mnist
from keras.datasets import cifar10
import keras
from keras.layers import Dense, Conv2D, BatchNormalization, Activation, Bidirectional, LSTM, Lambda, Concatenate, concatenate
from keras.layers import AveragePooling2D, Input, Flatten
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from keras import backend as K
from keras.datasets import cifar10
import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Reshape, Input, merge
from keras.layers import Conv2D, MaxPooling2D
import openml as oml
from keras.optimizers import SGD, RMSprop
from keras.models import load_model
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.convolutional import ZeroPadding2D, AveragePooling2D
from keras.models import Model
from keras.layers.advanced_activations import LeakyReLU
from keras.models import model_from_json
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from scipy.io import loadmat
from sklearn.decomposition import PCA

oml.config.apikey = 'b15b073c6fea6dc55b08f051f5e1abf9'
oml.config.cache_directory = os.path.expanduser('~/.openml/cache')
cachedir = '~/.openml/cache'
from scipy.io import loadmat
import scipy
from scipy import stats
import numpy as np
import inspect

excl=0 #0-8
T=20
pairdataamount=2500#100
valpairdatamount=240#20


#Get Task
#mnist_task=oml.tasks.get_task(3573)
#fmnist_task=oml.tasks.get_task(146825)
#emnist_task=oml.tasks.get_task(168294)
#usps_task=oml.tasks.get_task(168298)
#cifar10_task=oml.tasks.get_task(167124)
#cifar10small_task=oml.tasks.get_task(167133)
#svhn_task=task=oml.tasks.get_task(168297)
#olivettifaces_task=task=oml.tasks.get_task(168299)
#umistfacescropped_task=task=oml.tasks.get_task(168300)

#kdd_task=oml.tasks.get_task(3944)

#X1,y1=mnist_task.get_X_and_y()
#X2,y2=fmnist_task.get_X_and_y()
import random


mnist1=np.load('npy/2828_mnist_1.npy')#[:10,:]
mnist2=np.load('npy/2828_mnist_2.npy')#[:10,:]
mnist3=np.load('npy/2828_mnist_3.npy')#[:10,:]
mnist4=np.load('npy/2828_mnist_4.npy')#[:10,:]
mnist5=np.load('npy/2828_mnist_5.npy')#[:10,:]

fmnist1=np.load('npy/2828_fmnist_1.npy')#[:10,:]
fmnist2=np.load('npy/2828_fmnist_2.npy')#[:10,:]
fmnist3=np.load('npy/2828_fmnist_3.npy')#[:10,:]
fmnist4=np.load('npy/2828_fmnist_4.npy')#[:10,:]
fmnist5=np.load('npy/2828_fmnist_5.npy')#[:10,:]

emnist1=np.load('npy/2828_emnist_1.npy')#[:10,:]
emnist2=np.load('npy/2828_emnist_2.npy')#[:10,:]
emnist3=np.load('npy/2828_emnist_3.npy')#[:10,:]
emnist4=np.load('npy/2828_emnist_4.npy')#[:10,:]
emnist5=np.load('npy/2828_emnist_5.npy')#[:10,:]

cifar101=np.load('npy/2828_cifar10_1.npy')#[:10,:]
cifar102=np.load('npy/2828_cifar10_2.npy')#[:10,:]
cifar103=np.load('npy/2828_cifar10_3.npy')#[:10,:]
cifar104=np.load('npy/2828_cifar10_4.npy')#[:10,:]
cifar105=np.load('npy/2828_cifar10_5.npy')#[:10,:]

cifar10small1=np.load('npy/2828_cifar10small_1.npy')
cifar10small2=np.load('npy/2828_cifar10small_2.npy')
cifar10small3=np.load('npy/2828_cifar10small_3.npy')
cifar10small4=np.load('npy/2828_cifar10small_4.npy')
cifar10small5=np.load('npy/2828_cifar10small_5.npy')

olivettifaces1=np.load('npy/2828_olivettifaces_1.npy')
olivettifaces2=np.load('npy/2828_olivettifaces_2.npy')
olivettifaces3=np.load('npy/2828_olivettifaces_3.npy')
olivettifaces4=np.load('npy/2828_olivettifaces_4.npy')
olivettifaces5=np.load('npy/2828_olivettifaces_5.npy')

umistfacescropped1=np.load('npy/2828_umistfacescropped_1.npy')
umistfacescropped2=np.load('npy/2828_umistfacescropped_2.npy')
umistfacescropped3=np.load('npy/2828_umistfacescropped_3.npy')
umistfacescropped4=np.load('npy/2828_umistfacescropped_4.npy')
umistfacescropped5=np.load('npy/2828_umistfacescropped_5.npy')

svhn1=np.load('npy/2828_svhn_1.npy')
svhn2=np.load('npy/2828_svhn_2.npy')
svhn3=np.load('npy/2828_svhn_3.npy')
svhn4=np.load('npy/2828_svhn_4.npy')
svhn5=np.load('npy/2828_svhn_5.npy')

stl101=np.load('npy/2828_stl10_1.npy')
stl102=np.load('npy/2828_stl10_2.npy')
stl103=np.load('npy/2828_stl10_3.npy')
stl104=np.load('npy/2828_stl10_4.npy')
stl105=np.load('npy/2828_stl10_5.npy')


#new
t_mnist=np.array([0.9727,0.9762,0.9651,0.9737,0.922,0.9449,0.9809,0.9852,0.9624,0.9674,0.6309,0.9186,0.8085,0.9389,0.6423,0.674,0.4371,0.5198,0.9538,0.9553,0.8667,0.9184,0.242,0.1684,0.9617,0.9726,0.9632,0.9633,0.9601,0.962,0.5954,0.5759,0.9821,0.9853,0.4425,0.8454])
t_fmnist=np.array([0.8332,0.8478,0.7933,0.8265,0.727,0.7502,0.8626,0.8744,0.8302,0.846,0.7163,0.7874,0.8534,0.8112,0.5005,0.5109,0.4561,0.5052,0.821,0.8517,0.7576,0.8002,0.3993,0.2803,0.8432,0.8582,0.8742,0.8666,0.7925,0.867,0.8691,0.9025,0.8931,0.9123,0.7993,0.8565])
t_emnist=np.array([0.8079,0.8179,0.7823,0.8092,0.6368,0.7188,0.8294,0.8454,0.8187,0.8482,0.6615,0.8061,0.8295,0.8562,0.6019,0.7132,0.1501,0.2108,0.8208,0.8322,0.8124,0.8213,0.0437,0.2966,0.8135,0.8241,0.8199,0.8347,0.8515,0.8596,0.875,0.8786,0.8732,0.8841,0.877,0.8834])
t_cifar10=np.array([0.3446,0.3717,0.3346,0.4003,0.2415,0.2752,0.4201,0.4595,0.3644,0.4331,0.1429,0.1671,0.3089,0.3852,0.1998,0.2932,0.143,0.1854,0.4306,0.4843,0.4068,0.4319,0.1411,0.1387,0.1978,0.2367,0.4981,0.5314,0.2692,0.348,0.5292,0.5772,0.468,0.5568,0.231,0.2659])
t_cifar10small=np.array([0.3168,0.3499,0.2573,0.3219,0.1553,0.2024,0.2977,0.3634,0.1968,0.2821,0.1097,0.1276,0.3094,0.2316,0.2072,0.2142,0.1115,0.1227,0.3248,0.394,0.2091,0.3015,0.2154,0.2081,0.2066,0.2027,0.4054,0.4455,0.2515,0.3456,0.3487,0.4352,0.3087,0.4369,0.1481,0.214])
t_olivettifaces=np.array([0.025,0.025,0.025,0.025,0.025,0.025,0.025,0.025,0.025,0.025,0.025,0.025,0.1525,0.3175,0.0775,0.1325,0.0575,0.07,0.4225,0.4875,0.175,0.33,0.1025,0.16,0.075,0.1925,0.035,0.055,0.075,0.0475,0.0325,0.0375,0.03,0.035,0.025,0.04])
t_umistfaces=np.array([0.0748,0.08,0.0678,0.0765,0.0661,0.08,0.0696,0.0713,0.0643,0.0626,0.0765,0.0678,0.0643,0.0696,0.0609,0.0417,0.0696,0.0643,0.647,0.5791,0.2696,0.4452,0.1304,0.207,0.3322,0.5304,0.1043,0.1061,0.0643,0.06,0.0626,0.0696,0.0713,0.0626,0.0817,0.0661])
t_svhn=np.array([0.7314,0.6469,0.6386,0.6792,0.2696,0.39,0.8429,0.8737,0.7823,0.8174,0.2558,0.4713,0.5139,0.8334,0.6367,0.5868,0.1719,0.2038,0.7775,0.8335,0.7474,0.8249,0.2288,0.1848,0.4844,0.4027,0.8341,0.8349,0.7361,0.8553,0.3324,0.1867,0.6135,0.7744,0.4187,0.5714])
t_stl10=np.array([0.4438,0.4722,0.2858,0.3438,0.1,0.1216,0.3682,0.4418,0.1086,0.1913,0,0,0.1822,0.0999,0.1355,0.1629,0.1782,0.1646,0.4936,0.5388,0.4062,0.4857,0.3541,0.3392,0.2514,0.3091,0.2961,0.3893,0.2358,0.2774,0.1001,0.1,0.1135,0.1031,0.1351,0.1473])

#new
#t_mnist=np.array([0.9727,0.9762,0.9651,0.9737,0.922,0.9449,0.9809,0.9852,0.9624,0.9674,0.6309,0.9186,0.8085,0.9389,0.6423,0.674,0.4371,0.5198,0.9538,0.9553,0.8667,0.9184,0.242,0.1684,0.9617,0.9726,0.9632,0.9633,0.9601,0.962,0.5954,0.5759,0.9821,0.9853,0.4425,0.8454])
#t_fmnist=np.array([0.8332,0.8478,0.7933,0.8265,0.727,0.7502,0.8626,0.8744,0.8302,0.846,0.7163,0.7874,0.8534,0.8112,0.5005,0.5109,0.4561,0.5052,0.821,0.8517,0.7576,0.8002,0.3993,0.2803,0.8432,0.8582,0.8742,0.8666,0.7925,0.867,0.8691,0.9025,0.8931,0.9123,0.7993,0.8565])
#t_emnist=np.array([0.8079,0.8179,0.7823,0.8092,0.6368,0.7188,0.8294,0.8454,0.8187,0.8482,0.6615,0.8061,0.8295,0.8562,0.6019,0.7132,0.1501,0.2108,0.8208,0.8322,0.8124,0.8213,0.0437,0.2966,0.8135,0.8241,0.8199,0.8347,0.8515,0.8596,0.875,0.8786,0.8732,0.8841,0.877,0.8834])
#t_cifar10=np.array([0.3446,0.3717,0.3346,0.4003,0.2415,0.2752,0.4201,0.4595,0.3644,0.4331,0.1429,0.1671,0.3089,0.3852,0.1998,0.2932,0.143,0.1854,0.4306,0.4843,0.4068,0.4319,0.1411,0.1387,0.1978,0.2367,0.4981,0.5314,0.2692,0.348,0.5292,0.5772,0.468,0.5568,0.231,0.2659])
#t_cifar10small=np.array([0.3168,0.3499,0.2573,0.3219,0.1553,0.2024,0.2977,0.3634,0.1968,0.2821,0.1097,0.1276,0.3094,0.2316,0.2072,0.2142,0.1115,0.1227,0.3248,0.394,0.2091,0.3015,0.2154,0.2081,0.2066,0.2027,0.4054,0.4455,0.2515,0.3456,0.3487,0.4352,0.3087,0.4369,0.1481,0.214])
#t_olivettifaces=np.array([0.025,0.025,0.025,0.025,0.025,0.025,0.025,0.025,0.025,0.025,0.025,0.025,0.1525,0.3175,0.0775,0.1325,0.0575,0.07,0.4225,0.4875,0.175,0.33,0.1025,0.16,0.075,0.1925,0.035,0.055,0.075,0.0475,0.0325,0.0375,0.03,0.035,0.025,0.04])
#t_umistfaces=np.array([0.0748,0.08,0.0678,0.0765,0.0661,0.08,0.0696,0.0713,0.0643,0.0626,0.0765,0.0678,0.0643,0.0696,0.0609,0.0417,0.0696,0.0643,0.647,0.5791,0.2696,0.4452,0.1304,0.207,0.3322,0.5304,0.1043,0.1061,0.0643,0.06,0.0626,0.0696,0.0713,0.0626,0.0817,0.0661])
#t_svhn=np.array([0.7314,0.6469,0.6386,0.6792,0.2696,0.39,0.8429,0.8737,0.7823,0.8174,0.2558,0.4713,0.5139,0.8334,0.6367,0.5868,0.1719,0.2038,0.7775,0.8335,0.7474,0.8249,0.2288,0.1848,0.4844,0.4027,0.8341,0.8349,0.7361,0.8553,0.3324,0.1867,0.6135,0.7744,0.4187,0.5714])
#t_stl10=np.array([0.4438,0.4722,0.2858,0.3438,0.1,0.1216,0.3682,0.4418,0.1086,0.1913,0,0,0.1822,0.0999,0.1355,0.1629,0.1782,0.1646,0.4936,0.5388,0.4062,0.4857,0.3541,0.3392,0.2514,0.3091,0.2961,0.3893,0.2358,0.2774,0.1001,0.1,0.1135,0.1031,0.1351,0.1473])

def shuffle(a):
    _a=a
    np.random.shuffle(_a)
    return _a

def get_dataset(a):
    dataid=np.random.randint(1,6)
    #dataid=1
    if(a==0):
        if(dataid==1):
            return mnist1
        elif(dataid==2):
            return mnist2
        elif(dataid==3):
            return mnist3
        elif(dataid==4):
            return mnist4
        elif(dataid==5):
            return mnist5
    elif(a==1):
        if(dataid==1):
            return fmnist1
        elif(dataid==2):
            return fmnist2
        elif(dataid==3):
            return fmnist3
        elif(dataid==4):
            return fmnist4
        elif(dataid==5):
            return fmnist5
    elif(a==2):
        if(dataid==1):
            return emnist1
        elif(dataid==2):
            return emnist2
        elif(dataid==3):
            return emnist3
        elif(dataid==4):
            return emnist4
        elif(dataid==5):
            return emnist5
    elif(a==3):
        if(dataid==1):
            return cifar101
        elif(dataid==2):
            return cifar102
        elif(dataid==3):
            return cifar103
        elif(dataid==4):
            return cifar104
        elif(dataid==5):
            return cifar105
    elif(a==4):
        if(dataid==1):
            return cifar10small1
        elif(dataid==2):
            return cifar10small2
        elif(dataid==3):
            return cifar10small3
        elif(dataid==4):
            return cifar10small4
        elif(dataid==5):
            return cifar10small5
    elif(a==5):
        if(dataid==1):
            return olivettifaces1
        elif(dataid==2):
            return olivettifaces2
        elif(dataid==3):
            return olivettifaces3
        elif(dataid==4):
            return olivettifaces4
        elif(dataid==5):
            return olivettifaces5
    elif(a==6):
        if(dataid==1):
            return umistfacescropped1
        elif(dataid==2):
            return umistfacescropped2
        elif(dataid==3):
            return umistfacescropped3
        elif(dataid==4):
            return umistfacescropped4
        elif(dataid==5):
            return umistfacescropped5
    elif(a==7):
        if(dataid==1):
            return svhn1
        elif(dataid==2):
            return svhn2
        elif(dataid==3):
            return svhn3
        elif(dataid==4):
            return svhn4
        elif(dataid==5):
            return svhn5
    elif(a==8):
        if(dataid==1):
            return stl101
        elif(dataid==2):
            return stl102
        elif(dataid==3):
            return stl103
        elif(dataid==4):
            return stl104
        elif(dataid==5):
            return stl105

MNIST=0
FMNIST=1
EMNIST=2
CIFAR10=3
CIFAR10SMALL=4
OLIVETTIFACES=5
UMISTFACESCROPPED=6
SVHN=7
STL10=8

def get_shuffle(a):
    return shuffle(get_dataset(a))[:200,:]#get_dataset(a)[:200,:]

def get_pair(a,b):
    return get_shuffle(a),get_shuffle(b)

def make50(arr36):
    retarr=np.zeros(50)
    retarr[:len(arr36)]=arr36
    return retarr



bs=1
step=100
sample_size=100

def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.sum(K.square(x - y), axis=1, keepdims=True))


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)

def concat_output_shape(shapes):
    print(shapes)
    shape1, shape2 = shapes

def custom_loss(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    print('{} {}'.format(y_true,y_pred))
    return K.square((K.sum(K.abs(y_true[:,:50]-y_true[:,50:])))-(K.sqrt(K.sum(K.square(y_pred[:,:50]-y_pred[:,50:])))))

def custom_loss_tes(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    print('{} {}'.format(y_true,y_pred))
    y_true=K.reshape(y_true,(20,100))
    y_pred=K.reshape(y_pred,(20,100))

    #return K.sum(K.abs((K.sum(K.abs(y_true[:,:50]-y_true[:,50:]),axis=-1))-(K.sum(K.square(y_pred[:,:50]-y_pred[:,50:]),axis=-1))))
    return K.sum(K.square((K.sum(K.abs(y_true[:,:50]-y_true[:,50:]),axis=-1))-(K.sqrt(K.sum(K.square(y_pred[:,:50]-y_pred[:,50:]),axis=-1)))))

def custom_acc(y_true, y_pred):
    #return K.mean( K.abs((K.sum(K.abs(y_true[:,:50]-y_true[:,50:]),axis=-1))-(K.sum(K.abs(y_pred[:,:50]-y_pred[:,50:]),axis=-1)))<=1)
    return K.mean(K.square((K.sum(K.abs(y_true[:,:50]-y_true[:,50:]),axis=-1))-(K.sqrt(K.sum(K.square(y_pred[:,:50]-y_pred[:,50:]),axis=-1))))<=1)

def base_network(input_dim):
    input_shape=(input_dim,)
    model = Sequential()
    model.add(Reshape((28, 28,1), input_shape=(input_dim,)))
    #model.add(Flatten())
    model.add(Dense(300, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(300, activation='relu'))
    model.add(Dropout(0.5))
    #model.add(Flatten())
    #print(model.output_shape)
    model.add(Reshape((784, 300)))

    #model.add(LSTM(300,return_sequences=False))
    model.add(Bidirectional(LSTM(300,return_sequences=True)))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(50, activation='relu'))
    return model

def base_network_cnn(input_dim):
    input_shape=(input_dim,)
    model = Sequential()
    model.add(Reshape((28, 28,1), input_shape=(input_dim,)))
    #model.add(Flatten())
    model.add(Conv2D(32, (5,5), strides=(1,1), activation='relu',padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))
    model.add(Conv2D(64, (5,5), strides=(1,1), activation='relu',padding='same' ))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))
    model.add(Dense(1000, activation='relu'))
    #model.add(Dropout(0.2))
    model.add(Dense(300, activation='relu'))
    #model.add(Dropout(0.2))
    print(model.output_shape)
    model.add(Reshape((49, 300)))
    model.add(Bidirectional(LSTM(300,return_sequences=True)))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(50, activation='relu'))

    #print(model.output_shape)

    #model.add(LSTM(300,return_sequences=False))
    #model.summary()
    return model

def base_network_cnn_feature(input_dim):
    input_shape=(input_dim,)
    model = Sequential()
    model.add(Reshape((28, 28,1), input_shape=input_shape))
    #model.add(Flatten())
    model.add(Conv2D(32, (5,5), strides=(1,1), activation='relu',padding='valid'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (5,5), strides=(1,1), activation='relu',padding='valid' ))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    #model.add(Dropout(0.2))
    model.add(Dense(1000, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(300, activation='relu'))
    model.add(Dropout(0.2))

    #print(model.output_shape)


    #model.summary()
    return model

rms = RMSprop(lr=0.0001)#(lr=0.0001)
L1_distance = lambda x: K.abs(x[0]-x[1])

def siam1():
    basenet=base_network_cnn(784)

    input_a = Input(shape=(784,))
    input_b = Input(shape=(784,))

    processed_a = basenet(input_a)
    processed_b = basenet(input_b)

    #distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([processed_a, processed_b])
    concalambda = (concatenate)([processed_a, processed_b])

    #conca= Concatenate([processed_a,processed_b])
    model = Model(inputs=[input_a, input_b], outputs=concalambda)

    #merge two encoded inputs with the l1 distance between them
    ##L1_distance = lambda x: K.abs(x[0]-x[1])
    #prediction = Dense(1,activation='sigmoid',bias_initializer=b_init)(both)
    ##both = merge([left_wing,right_wing],
    ##             mode = L1_distance,
    ##             output_shape=lambda x: x[0])
    ##prediction = Dense(10,activation='sigmoid')(both)
    #fc3=Dense(100)(prediction)
    #fc4=Dense(50)(fc3)
    ##siamese_net = Model(input=[left_wing,right_wing],output=prediction)
    #siamese_net.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=["accuracy"])
    model.compile(loss=custom_loss, optimizer=keras.optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.))
    return model#siamese_net

def lstm_model(sz):
    y=Sequential()
    y.add(Reshape((sz,300),input_shape=(sz*300,)))
    y.add(Bidirectional(LSTM(300,return_sequences=False)))
    y.add(Dense(100, activation='relu'))
    y.add(Dense(50))
    return y

def siam2():
    basenet=base_network_cnn_feature(784)
    sz=200
    input_a = Input(shape=(sz,784))
    input_b = Input(shape=(sz,784))
    a=[{} for i in range(sz)]
    processed_a=[{} for i in range(sz)]
    for i in range(sz):
        a[i]=Lambda(lambda x: x[:,i],name='concat_a{}'.format(i))(input_a)
        processed_a[i] = basenet(a[i])

    testa=concatenate(processed_a,axis=-1)
    #x=Reshape((sz,300))(testa)
    #x=Bidirectional(LSTM(300,return_sequences=False))(x)
    #x=Dense(100, activation='relu')(x)
    #x=Dense(50, activation='relu')(x)
    lstmnet=lstm_model(sz)
    x=lstmnet(testa)

    b=[{} for i in range(sz)]
    processed_b=[{} for i in range(sz)]
    for i in range(sz):
        b[i]=Lambda(lambda x: x[:,i],name='concat_b{}'.format(i))(input_b)
        processed_b[i] = basenet(b[i])

    testb=concatenate(processed_b,axis=-1)
    #y=Reshape((sz,300))(testb)
    #y=Bidirectional(LSTM(300,return_sequences=False))(y)
    #y=Dense(100, activation='relu')(y)
    #y=Dense(50, activation='relu')(y)
    y=lstmnet(testb)
    end=concatenate([x,y])
    #conca= Concatenate([processed_a,processed_b])
    #model = Model(inputs=[input_a, input_b], outputs=concalambda)
    model = Model(inputs=[input_a, input_b], outputs=end)
    #model.summary()

    model.compile(loss=custom_loss_tes, optimizer=rms, metrics=[custom_acc])
    return model#siamese_net


#sample_size=100
#rand1=random.sample(range(len(X1)), sample_size)
#rand2=random.sample(range(len(X2)), sample_size)

def l1_dist(a,b):
    return np.sum([np.abs(a-b) for item in a])

def l2_dist(a,b):
    return np.sqrt(np.sum([np.square(a-b) for item in a]))


model=siam2()

"""
y1=np.append(make50(t_mnist),make50(t_fmnist))
y2=np.append(make50(t_fmnist),make50(t_mnist))
y=np.zeros((10000,100))
for i in range(5000):
    y[i]=y1
for i in range(5000,10000):
    y[i]=y2
X1,X2=get_pair(0,1)
Xleft=np.append(X1,X2,axis=0)
Xright=np.append(X2,X1,axis=0)
"""



def get_t(dataid):
    if(dataid==0):
        return t_mnist
    elif(dataid==1):
        return t_fmnist
    elif(dataid==2):
        return t_emnist
    elif(dataid==3):
        return t_cifar10
    elif(dataid==4):
        return t_cifar10small
    elif(dataid==5):
        return t_olivettifaces
    elif(dataid==6):
        return t_umistfaces
    elif(dataid==7):
        return t_svhn
    elif(dataid==8):
        return t_stl10

def make_y(id,pair1,pair2):
    global y
    y[id,:50]=make50(get_t(pair1))
    y[id,50:]=make50(get_t(pair2))
def make_yval(id,pair1,pair2):
    global yval
    yval[id,:50]=make50(get_t(pair1))
    yval[id,50:]=make50(get_t(pair2))
    #return y

'''
y=np.zeros((6,100))
y[0,:50]=make50(t_mnist)
y[0,50:]=make50(t_fmnist)
y[1,:50]=make50(t_mnist)
y[1,50:]=make50(t_emnist)
y[2,:50]=make50(t_mnist)
y[2,50:]=make50(t_cifar10)
y[3,:50]=make50(t_fmnist)
y[3,50:]=make50(t_emnist)
y[4,:50]=make50(t_fmnist)
y[4,50:]=make50(t_cifar10)
y[5,:50]=make50(t_emnist)
y[5,50:]=make50(t_cifar10)
'''
print('Excluding: '+str(excl))
y=np.zeros((pairdataamount,100))
yval=np.zeros((valpairdatamount,100))

for i in range(T):
    #id=[(i)%9 for i in range(2*pairdataamount+((2*pairdataamount)//9)+40) if (i%9)!=excl]
    id=[(i)%9 for i in range(3*pairdataamount) if (i%9)!=excl]
    id=shuffle(id)
    print(len(id))
    #print(id)
    #id=[0,1,2,3,4,5,6,7]
    Xleft=np.zeros((pairdataamount,200,784),dtype=float)
    Xright=np.zeros((pairdataamount,200,784),dtype=float)

    for ii in range(pairdataamount):
        make_y(ii,id[2*ii],id[2*ii+1])
        Xleft[ii],Xright[ii]=get_pair(id[2*ii],id[2*ii+1])

    #X3,X4=get_pair(id[2],id[3])
    #X5,X6=get_pair(id[4],id[5])
    #X7,X8=get_pair(id[6],id[7])
    ###yval=np.zeros((1,100))
    ###y_val[0,:50]=make50(get_t(id[]))
    #Xleft=np.concatenate([[X1],[X3],[X5]],axis=0)
    #Xright=np.concatenate([[X2],[X4],[X6]],axis=0)
    Xvalleft=np.zeros((valpairdatamount,200,784),dtype=float)
    Xvalright=np.zeros((valpairdatamount,200,784),dtype=float)
    for ii in range(valpairdatamount):
        make_yval(ii,excl,(ii)%9)
        Xvalleft[ii],Xvalright[ii]=get_pair(excl,ii%9)

    #model.fit([Xleft, Xright], y[:3,:],validation_data=([[X7],[X8]],y[3:,:]),batch_size=1,epochs=10,verbose=2)
    #model.fit([Xleft[:80,:], Xright[:80,:]], y[:80,:],validation_data=([Xleft[80:,:],Xright[80:,:]],y[80:,:]),batch_size=20,epochs=1,verbose=2)
    model.fit([Xleft, Xright], y,validation_data=([Xvalleft,Xvalright], yval),batch_size=20,epochs=1,verbose=2)


Xl0,Xr0=get_pair(0,0)
Xl1,Xr1=get_pair(1,1)
Xl2,Xr2=get_pair(2,2)
Xl3,Xr3=get_pair(3,3)
Xl4,Xr4=get_pair(4,4)
Xl5,Xr5=get_pair(5,5)
Xl6,Xr6=get_pair(6,6)
Xl7,Xr7=get_pair(7,7)
Xl8,Xr8=get_pair(8,8)

Xl9,Xr9=get_pair(0,0)
Xl10,Xr10=get_pair(1,1)
Xl11,Xr11=get_pair(2,2)
Xl12,Xr12=get_pair(3,3)
Xl13,Xr13=get_pair(4,4)
Xl14,Xr14=get_pair(5,5)
Xl15,Xr15=get_pair(6,6)
Xl16,Xr16=get_pair(7,7)
Xl17,Xr17=get_pair(8,8)


Xleft=np.concatenate([[Xl0],[Xl1],[Xl2],[Xl3],[Xl4],[Xl5],[Xl6],[Xl7],[Xl8],[Xl9],[Xl10],[Xl11],[Xl12],[Xl13],[Xl14],[Xl15],[Xl16],[Xl17]],axis=0)
Xright=np.concatenate([[Xr0],[Xr1],[Xr2],[Xr3],[Xr4],[Xr5],[Xr6],[Xr7],[Xr8],[Xr9],[Xr10],[Xr11],[Xr12],[Xr13],[Xr14],[Xr15],[Xr16],[Xr17]],axis=0)

model.save_weights('siam_weights_'+str(excl)+'_'+str(pairdataamount)+'p_'+str(T)+'i.h5')
json_model = model.to_json()
open('siam_arch_'+str(excl)+'_'+str(pairdataamount)+'p_'+str(T)+'i.json', 'w').write(json_model)
prediction=model.predict([Xleft,Xright])
prediction.dump('siam_prediction_'+str(excl)+'_'+str(pairdataamount)+'p_'+str(T)+'i')
