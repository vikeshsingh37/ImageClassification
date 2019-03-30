#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
#os.environ["CUDA_VISIBLE_DEVICES"]="0" 
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.client import device_lib
import seaborn as sns
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D,Dropout,Dense, Flatten, Reshape
from keras import optimizers
from IPython.display import clear_output
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling2D
from keras.layers.core import Activation
from keras.optimizers import Adam
from tqdm import tqdm
from sklearn.metrics import accuracy_score
   
print(device_lib.list_local_devices())



# Check how training data looks like
train_dat = pd.read_csv("D:/Kaggle/digit-recognizer/train.csv")
train_dat.head()



y= train_dat["label"]
X = train_dat.drop("label", axis = 1)


# Data Analysis
sns.countplot(y)
X.isnull().any().any()


# There are no missing values and the labels 1 to 10 are uniformly distributed
# Now let's see what actual data looks like
# First reshape the 1D array to 28*28 pixel of gray scale
X = X.values.reshape(-1,28,28,1)
plt.imshow(X[80][:,:,0], cmap='gray')
y[80]


# In[6]:


# Normalize the data so that CNN can converge faster
X= X/255


# Since this is a classification problem, convert to categorical
y = to_categorical(y, num_classes= 11)
y.shape


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.1, random_state=42)


def discriminator():
    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size= (5,5), strides=1, padding='Same',activation ='relu', input_shape = (28,28,1)))
    model.add(MaxPool2D(pool_size=(4,4)))
    model.add(Dropout(0.3))
    model.add(Conv2D(filters=16, kernel_size= (5,5), strides=1, padding='Same',activation ='relu'))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(12, activation = 'softmax'))
    return(model)


def generator():
    model = Sequential()
    model.add(Dense(input_dim=100, units=1024))
    model.add(Activation('relu'))
    model.add(Dense(128*7*7))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Reshape((7, 7, 128), input_shape=(128*7*7,)))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Conv2D(64, (5, 5), padding='same',activation ='relu'))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Conv2D(1, (5, 5), padding='same',activation ='tanh'))
    return model



def gan(g, d):
    model = Sequential()
    model.add(g)
    d.trainable = False
    model.add(d)
    optimizer = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer,metrics=["accuracy"])
    return model



dis_model = discriminator()
#dis_model.add(Dense(1, activation='sigmoid'))
optimizer = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
dis_model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])

gen_model = generator()
d_on_g = gan(gen_model, dis_model)

noise = np.random.uniform(-1, 1, size=[500, 100])
gen_img =gen_model.predict_on_batch(noise)
plt.imshow(gen_img[0][:,:,0],cmap='gray')


class Plotacc(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.acc = []
        self.val_acc = []
        
        self.fig = plt.figure()
        
        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        
        self.logs.append(logs)
        self.x.append(self.i)
        self.acc.append(logs.get('acc'))
        self.val_acc.append(logs.get('val_acc'))
        self.i += 1
        
        clear_output(wait=True)
        plt.plot(self.x, self.acc, label= 'acc:' + str(self.acc[self.i-1]))
        plt.plot(self.x, self.val_acc, label='val_acc:' + str(self.val_acc[self.i-1]))
        plt.legend()
        plt.show()
        
plot_acc = Plotacc()



batch_size = 1024*2
batch_count = int( X_train.shape[0]/batch_size)
#noise = np.random.uniform(-1, 1, size=[int(batch_size/1), 100])
for ep in range(100):
    for _ in tqdm(range(batch_count)):
        print('The epoch is %d'%ep)
        idx = np.random.randint(0, X_train.shape[0], batch_size)
        X_train_ep =  X_train[idx]
        y_train_ep = y_train[idx,:]
        # Add column for fake or not 
        y_train_ep = np.concatenate((y_train_ep,np.zeros(batch_size).reshape(batch_size,1)), axis = 1)
        #y_train_ep = np.ones(batch_size)
        
        # Add noise data
        noise = np.random.uniform(-1, 1, size=[int(batch_size), 100])
        noise_train_x = gen_model.predict_on_batch(noise)
        noise_train_y = np.repeat(np.array([0,0,0,0,0,0,0,0,0,0,0]), batch_size).reshape(batch_size,11) 
        noise_train_y = np.concatenate((noise_train_y,np.ones(batch_size).reshape(batch_size,1)),axis = 1)

        X_train_ep_all = np.concatenate((X_train_ep,noise_train_x))
        y_train_ep_all = np.concatenate((y_train_ep,noise_train_y))
        dis_model.trainable  = True
        y_val_ep = np.concatenate((y_val, np.zeros(y_val.shape[0]).reshape(y_val.shape[0],1)),axis = 1)
        dis_model.fit(X_train_ep_all, y_train_ep_all, validation_data = (X_val, y_val_ep),epochs=1,verbose = 0, callbacks=[plot_acc])
        
        # GAN training
        y_gan = d_on_g.predict_on_batch(noise)
        dis_model.trainable  = False
        noise_train_y[:,11] = np.zeros(batch_size)
        noise_train_y[:,np.random.randint(0,11)]=np.ones(batch_size)
        d_on_g.fit(noise,noise_train_y,epochs=1, verbose = 1)


        gen_img =gen_model.predict_on_batch(noise)
        plt.imshow(gen_img[0][:,:,0],cmap='gray')

#Use discriminator for predictions
X_test = pd.read_csv("D:/Kaggle/digit-recognizer/test.csv")
X_test = X_test.values.reshape(-1,28,28,1)
X_test = X_test/255


y_test = dis_model.predict(X_test)
y_test = np.argmax(y_test,axis = 1)
y_test = pd.Series(y_test,name="Label")
final_output = pd.concat([pd.Series(range(1,28001),name = "ImageId"),y_test],axis = 1)


final_output.to_csv("D:/Kaggle/digit-recognizer/submission.csv", index= False)

