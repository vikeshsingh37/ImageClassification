#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
   
print(device_lib.list_local_devices())


# In[2]:


# Check how training data looks like
train_dat = pd.read_csv("D:/Kaggle/digit-recognizer/train.csv")
train_dat.head()


# In[3]:


y= train_dat["label"]
X = train_dat.drop("label", axis = 1)


# In[4]:


# Data Analysis
sns.countplot(y)
X.isnull().any().any()


# In[5]:


# There are no missing values and the labels 1 to 10 are uniformly distributed
# Now let's see what actual data looks like
# First reshape the 1D array to 28*28 pixel of gray scale
X = X.values.reshape(-1,28,28,1)
plt.imshow(X[8][:,:,0])
y[8]


# In[6]:


# Normalize the data so that CNN can converge faster
X= X/255


# In[7]:


# Since this is a classification problem, convert to categorical
y = to_categorical(y, num_classes= 11)
y.shape


# In[8]:


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.1, random_state=42)


# In[9]:


def discriminator():
    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size= (5,5), strides=1, padding='Same',activation ='relu', input_shape = (28,28,1)))
    model.add(MaxPool2D(pool_size=(4,4)))
    model.add(Dropout(0.3))
    model.add(Conv2D(filters=16, kernel_size= (5,5), strides=1, padding='Same',activation ='relu'))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(11, activation = 'softmax'))
    return(model)


# In[10]:


def generator():
    model = Sequential()
    model.add(Dense(input_dim=100, output_dim=1024))
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


# In[11]:


gen_model = generator()
optimizer = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
gen_model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])


# In[12]:


dis_model = discriminator()
optimizer = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
dis_model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])


# In[13]:


def gan(g, d):
    model = Sequential()
    model.add(g)
    d.trainable = False
    model.add(d)
    return model


# In[14]:


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


# In[15]:


noise = []
for i in range(500):
    noise.append(np.random.uniform(-1, 1, (1, 100)))


# In[16]:


for ep in range(50):
    X_train_ep = X_train
    y_train_ep = y_train
    print(ep)
    for i in range(500):
        noise_train_x = gen_model.predict(noise[i])
        noise_train_y = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.]).reshape(1,11)

        X_train_ep = np.concatenate((X_train_ep,noise_train_x))
        y_train_ep = np.concatenate((y_train_ep,noise_train_y))
    
    dis_model.fit(X_train_ep, y_train_ep, epochs=30, batch_size=80,validation_data = (X_val, y_val),  callbacks=[plot_acc], verbose=0)
    
    d_on_g = gan(gen_model, dis_model)
    d_on_g.compile(loss='binary_crossentropy', optimizer=optimizer)
    
    # GAN training
    y_gan = []
    for i in range(500):
        y_gan.append(d_on_g.predict(noise[i]))
        replace_i = np.random.randint(low=0,high=9,size=1).tolist()[0]
        #predicted_value = d_on_g.predict(noise[i])

        # Make GAN to learn not to predict 11th element which is noise correctly
        y_gan[i][:,0:10] = np.array(0)
        y_gan[i][:,replace_i] = 1
        d_on_g.fit(noise[i],y_gan[i],epochs=10, verbose = 0)
    
    dis_model.trainable = True
    
    gen_img =gen_model.predict(noise[np.random.randint(low=0,high=99,size=1).tolist()[0]])
    plt.imshow(gen_img[0][:,:,0])


# In[17]:


X_test = pd.read_csv("D:/Kaggle/digit-recognizer/test.csv")
X_test = X_test.values.reshape(-1,28,28,1)
X_test = X_test/255


# In[18]:


y_test = dis_model.predict(X_test)
y_test = np.argmax(y_test,axis = 1)
y_test = pd.Series(y_test,name="Label")
final_output = pd.concat([pd.Series(range(1,28001),name = "ImageId"),y_test],axis = 1)


# In[19]:


final_output.to_csv("D:/Kaggle/digit-recognizer/submission.csv", index= False)


# In[20]:


noise = np.random.uniform(0, 10, size=(1, 100)).reshape(1,100)
gen_img =gen_model.predict(noise)
plt.imshow(gen_img[0][:,:,0])


# In[ ]:




