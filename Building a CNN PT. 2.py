#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import keras
from keras.datasets import mnist
from keras.models import Model
from keras.layers import Dense, Input
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten
from keras import backend as k


# 3. Build and train a Keras CNN classifier on the MNIST training set.
# 
# Using this link for reference: https://www.geeksforgeeks.org/applying-convolutional-neural-network-on-mnist-dataset/

# In[2]:


(x_train, y_train), (x_test, y_test) = mnist.load_data()


# In[3]:


img_rows, img_cols=28, 28
 
if k.image_data_format() == 'channels_first':
   x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
   x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
   inpx = (1, img_rows, img_cols)
 
else:
   x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
   x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
   inpx = (img_rows, img_cols, 1)
 
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255


# In[4]:


y_train = keras.utils.to_categorical(y_train)
y_test = keras.utils.to_categorical(y_test)


# In[5]:


inpx = Input(shape=inpx)
layer1 = Conv2D(32, kernel_size=(3, 3), activation='relu')(inpx)
layer2 = Conv2D(64, (3, 3), activation='relu')(layer1)
layer3 = MaxPooling2D(pool_size=(3, 3))(layer2)
layer4 = Dropout(0.5)(layer3)
layer5 = Flatten()(layer4)
layer6 = Dense(250, activation='sigmoid')(layer5)
layer7 = Dense(10, activation='softmax')(layer6)


# In[6]:


model = Model([inpx], layer7)
model.compile(optimizer=keras.optimizers.Adadelta(),
              loss=keras.losses.categorical_crossentropy,
              metrics=['accuracy'])
 
model.fit(x_train, y_train, epochs=12, batch_size=500)


# 4. Report the test accuracy of your model.

# In[7]:


score = model.evaluate(x_test, y_test, verbose=0)
print('Accuracy: ', score[1])


# 5. Display a confusion matrix on the test set classifications.

# In[15]:


from sklearn.metrics import confusion_matrix
from sklearn import metrics


# In[16]:


y_pred = model.predict(x_test)
matrix = metrics.confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))


# In[17]:


print(matrix)


# 6. Summarize the results.
# 

# After 12 epochs, the accuracy of the model was reported to be about 25%, which isn't great. The losses were around 2 during each epoch, and we would ideally like to have that number closer to 0. All in all, the model that was built and used today did not fit the data well.

# In[ ]:




