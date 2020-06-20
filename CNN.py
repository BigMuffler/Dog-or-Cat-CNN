# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 13:18:22 2020

@author: Stefan
"""
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from keras.preprocessing import image

# Preprocess Training Set
# Apply transormations to training set to avoid overfitting 
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
training_set = train_datagen.flow_from_directory(
        'dataset/training_set',
        target_size=(64, 64), #final size of images after being fed into CNN
        batch_size=32, #number of images in each batch batch
        class_mode='binary') #output is either cat or dog hence binary

test_datagen = ImageDataGenerator(rescale=1./255)
test_set = test_datagen.flow_from_directory(
        'dataset/test_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary') #output is either cat or dog hence binary

#Build CNN
cnn = tf.keras.models.Sequential()
#Convolution
#Each filter is randomly generated via the keras library
cnn.add(tf.keras.layers.Conv2D(filters = 32, kernel_size = 3, activation = 'relu', input_shape = [64,64,3]))
#Max Pooling
cnn.add(tf.keras.layers.MaxPool2D(pool_size = 2, strides = 2))
#Add second convolutional layer
cnn.add(tf.keras.layers.Conv2D(filters = 32, kernel_size = 3, activation = 'relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size = 2, strides = 2))

#Flattening the Result of the Convolutions to a 1-D Vector
#Essentially converts each pooled feature map matrix into a 1-D Vector
cnn.add(tf.keras.layers.Flatten())

#Making the full connection
cnn.add(tf.keras.layers.Dense(units=128, activation ='relu'))
#OuputLayer
cnn.add(tf.keras.layers.Dense(units=1, activation ='sigmoid'))

#Training the CNN
cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
cnn.fit(x = training_set, validation_data = test_set, epochs = 100)

#Making Predictions
test_image = image.load_img('dataset/single_prediction/cat2.jpg', target_size = (64,64))
#Convert PIL Image format to array 
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0) #something to do with batch size
result = cnn.predict(test_image)
training_set.class_indices
if result[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'

print(prediction)
