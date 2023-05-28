#!/usr/bin/env python
# coding: utf-8

# In[74]:


import numpy as np
import keras
from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Input
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras.optimizers import SGD
from keras import backend as K


# In this exercise, you will build a convolutional neural network (CNN) to classify handwritten digits from the MNIST dataset. The steps to build a CNN classifier are outlined in section 20.15 of the Machine Learning with Python Cookbook, but keep in mind that your code may need to be modified depending on your version of Keras.

# In[51]:


# Set that the color channel value will be first
K.set_image_data_format("channels_first")

# Set seed
np.random.seed(0)

# Set image information
channels = 1
height = 28
width = 28


# 1. Load the MNIST data set.

# 2. Display the first five images in the training data set (see section 8.1 in the Machine Learning with Python Cookbook). Compare these to the first five training labels.

# I am using this article plus the book for the code: https://python-course.eu/machine-learning/training-and-testing-with-mnist.php

# In[52]:


pip install opencv-python


# In[53]:


import cv2
import numpy as np
from matplotlib import pyplot as plt


# In[54]:


image_size = 28 # width and length
no_of_different_labels = 10 #  i.e. 0, 1, 2, 3, ..., 9
image_pixels = image_size * image_size
train_data = np.loadtxt("/Users/SophiaWeidner/Downloads/mnist_train.csv", delimiter = ',')
test_data = np.loadtxt("/Users/SophiaWeidner/Downloads/mnist_test.csv", delimiter = ',') 
test_data[:10]


# In[55]:


fac = 0.99 / 255
train_imgs = np.asfarray(train_data[:, 1:]) * fac + 0.01
test_imgs = np.asfarray(test_data[:, 1:]) * fac + 0.01

train_labels = np.asfarray(train_data[:, :1])
test_labels = np.asfarray(test_data[:, :1])


# In[56]:


import numpy as np

lr = np.arange(10)

# Printing first 5 labels.
for label in range(5):
    one_hot = (lr==label).astype(np.int)
    print("label: ", label, " in one-hot representation: ", one_hot)
    
# In the link I used to help structure this code, they went on to change the 0s and 1s to .01 and .99 as they
# said it would be better for calculations. I am going to hold off on this for now.


# In[57]:


# Printing first 5 images.
for i in range(5):
    img = train_imgs[i].reshape((28,28))
    plt.imshow(img, cmap="Greys")
    plt.show()


# 3. Build and train a Keras CNN classifier on the MNIST training set.
# 
# Using this link for reference: https://www.geeksforgeeks.org/applying-convolutional-neural-network-on-mnist-dataset/
# 
# I was having issues with this file and the model for whatever reason. I was able to fit the model appropriately in another jupyter notebook. I will continue the assignment in that new notebook.

# In[ ]:




