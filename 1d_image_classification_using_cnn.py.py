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
from keras.layers import Conv2D, MaxPool2D,Dropout,Dense, Flatten
from keras import optimizers
from IPython.display import clear_output
   
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
y = to_categorical(y, num_classes= 10)
y.shape


# In[8]:


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.1, random_state=42)


# In[9]:


model = Sequential()
model.add(Conv2D(filters=32, kernel_size= (5,5), strides=1, padding='Same',activation ='relu', input_shape = (28,28,1)))
model.add(MaxPool2D(pool_size=(4,4)))
model.add(Dropout(0.3))
model.add(Conv2D(filters=16, kernel_size= (5,5), strides=1, padding='Same',activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(10, activation = 'softmax'))
optimizer = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)


# In[10]:


model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])


# In[11]:


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
      


# In[12]:


model.fit(X_train, y_train, epochs=500, batch_size=80,validation_data = (X_val, y_val), callbacks=[plot_acc], verbose=0)


# In[13]:


X_test = pd.read_csv("D:/Kaggle/digit-recognizer/test.csv")
X_test = X_test.values.reshape(-1,28,28,1)
X_test = X_test/255


# In[14]:


y_test = model.predict(X_test)
y_test = np.argmax(y_test,axis = 1)
y_test = pd.Series(y_test,name="Label")
final_output = pd.concat([pd.Series(range(1,28001),name = "ImageId"),y_test],axis = 1)


# In[15]:


final_output.to_csv("D:/Kaggle/digit-recognizer/submission.csv", index= False)

