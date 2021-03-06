# -*- coding: utf-8 -*-
"""Image Classification model

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1uVmmtwzJjmWxF7p8rTBTHVkIZ2W6JDtR

# Introduction
* Convolutional neural networks (CNN) – the concept behind recent breakthroughs  and developments in deep learning.

CNNs have broken the mold and ascended the throne to become the state-of-the-art computer vision technique. Among the different types of neural networks (others include recurrent neural networks (RNN), long short term memory (LSTM), artificial neural networks (ANN), etc.), CNNs are easily the most popular.

There are various datasets that you can leverage for applying convolutional neural networks. Here are three popular datasets:

* MNIST
* CIFAR-10
* ImageNet

# Using CNNs to Classify Hand-written Digits on MNIST Dataset
###  MNIST CNN
![](https://cdn.analyticsvidhya.com/wp-content/uploads/2020/02/mnist.png)

---



MNIST (Modified National Institute of Standards and Technology) is a well-known dataset used in Computer Vision that was built by Yann Le Cun et. al. It is composed of images that are handwritten digits (0-9), split into a training set of 50,000 images and a test set of 10,000 where each image is of 28 x 28 pixels in width and height.
"""

# keras imports for the dataset and building our neural network
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPool2D, Flatten
from keras.utils import np_utils

# to calculate accuracy
from sklearn.metrics import accuracy_score

# loading the dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# building the input vector from the 28x28 pixels
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# normalizing the data to help with the training
X_train /= 255
X_test /= 255

# one-hot encoding using keras' numpy-related utilities
n_classes = 10
print("Shape before one-hot encoding: ", y_train.shape)
Y_train = np_utils.to_categorical(y_train, n_classes)
Y_test = np_utils.to_categorical(y_test, n_classes)
print("Shape after one-hot encoding: ", Y_train.shape)

# building a linear stack of layers with the sequential model
model = Sequential()
# convolutional layer
model.add(Conv2D(25, kernel_size=(3,3), strides=(1,1), padding='valid', activation='relu', input_shape=(28,28,1)))
model.add(MaxPool2D(pool_size=(1,1)))
# flatten output of conv
model.add(Flatten())
# hidden layer
model.add(Dense(100, activation='relu'))
# output layer
model.add(Dense(10, activation='softmax'))

# compiling the sequential model
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

# training the model for 10 epochs
model.fit(X_train, Y_train, batch_size=128, epochs=10, validation_data=(X_test, Y_test))

"""# Identifying Images from the CIFAR-10 Dataset using CNNs
![](https://cdn.analyticsvidhya.com/wp-content/uploads/2020/02/1_sGochNLZ-qfesdyjadgXNw.png)

The important points that distinguish this dataset from MNIST are:



Images are colored in CIFAR-10 as compared to the black and white texture of MNIST

* Each image is 32 x 32 colour pixel/images in 10 classes,
* 50,000 training images and 
* 10,000 testing images.

Now, these images are taken in varying lighting conditions and at different angles, and since these are colored images, you will see that there are many variations in the color itself of similar objects (for example, the color of ocean water). If you use the simple CNN architecture that we saw in the MNIST example above, you will get a low validation accuracy of around 60%.
"""

from keras.datasets import cifar10
# loading the dataset 
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# # building the input vector from the 32x32 pixels
X_train = X_train.reshape(X_train.shape[0], 32, 32, 3)
X_test = X_test.reshape(X_test.shape[0], 32, 32, 3)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# normalizing the data to help with the training
X_train /= 255
X_test /= 255

# one-hot encoding using keras' numpy-related utilities
n_classes = 10
print("Shape before one-hot encoding: ", y_train.shape)
Y_train = np_utils.to_categorical(y_train, n_classes)
Y_test = np_utils.to_categorical(y_test, n_classes)
print("Shape after one-hot encoding: ", Y_train.shape)

"""Here’s what I had to changed in the model:
* Increased the number of Conv2D layers to build a deeper model
* Increased number of filters to learn more features
* Added Dropout for regularization
* Added more Dense layers

Training and validation accuracy across epochs:
"""

# building a linear stack of layers with the sequential model
model = Sequential()

# convolutional layer
model.add(Conv2D(50, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu', input_shape=(32, 32, 3)))

# convolutional layer
model.add(Conv2D(75, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(125, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))

# flatten output of conv
model.add(Flatten())

# hidden layer
model.add(Dense(500, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(250, activation='relu'))
model.add(Dropout(0.3))
# output layer
model.add(Dense(10, activation='softmax'))

# compiling the sequential model
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

# training the model for 10 epochs
model.fit(X_train, Y_train, batch_size=128, epochs=10, validation_data=(X_test, Y_test))

"""# Categorizing the Images of ImageNet using CNNs

![](https://cdn.analyticsvidhya.com/wp-content/uploads/2020/02/ImageNet-Title-Pic.jpg)

ImageNet is the main database behind the ImageNet Large Scale Recognition Challenge (ILSVRC). This is like the Olympics of Computer Vision. This is the competition that made CNNs popular the first time and every year, the best research teams across industries and academia compete with their best algorithms on computer vision tasks.

 

About the ImageNet Dataset
The ImageNet dataset has more than 14 million images, hand-labeled across 20,000 categories.

Also, unlike the MNIST and CIFAR-10 datasets that we have already discussed, the images in ImageNet are of decent resolution (224 x 224) and that’s what poses a challenge for us: 14 million images, each 224 by 224 pixels. Processing a dataset of this size requires a great amount of computing power in terms of CPU, GPU, and RAM.

The downside – that might be too much for an everyday laptop. So what’s the alternative solution? How can an enthusiast work with the ImageNet dataset?

 

That’s where Fast.ai’s Imagenette dataset comes in
Imagenette is a dataset that’s extracted from the large ImageNet collection of images. The reason behind releasing Imagenette is that researchers and students can practice on ImageNet level images without needing that much compute resources.
"""

!wget https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz

!tar -xf imagenette2.tgz

"""2. Loading Images using ImageDataGenerator
Keras has this useful functionality for loading large images (like we have here) without maxing out the RAM, by doing it in small batches. ImageDataGenerator in combination with fit_generator provides this functionality:
"""

from keras.preprocessing.image import ImageDataGenerator

# create a new generator
imagegen = ImageDataGenerator()
# load train data
train = imagegen.flow_from_directory("imagenette2/train/", class_mode="categorical", shuffle=False, batch_size=128, target_size=(224, 224))
# load val data
val = imagegen.flow_from_directory("imagenette2/val/", class_mode="categorical", shuffle=False, batch_size=128, target_size=(224, 224))

"""3. Building a Basic CNN model for Image Classification

---


Let’s build a basic CNN model for our Imagenette dataset (for the purpose of image classification):
"""

from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, InputLayer, BatchNormalization, Dropout

# build a sequential model
model = Sequential()
model.add(InputLayer(input_shape=(224, 224, 3)))

# 1st conv block
model.add(Conv2D(25, (5, 5), activation='relu', strides=(1, 1), padding='same'))
model.add(MaxPool2D(pool_size=(2, 2), padding='same'))
# 2nd conv block
model.add(Conv2D(50, (5, 5), activation='relu', strides=(2, 2), padding='same'))
model.add(MaxPool2D(pool_size=(2, 2), padding='same'))
model.add(BatchNormalization())
# 3rd conv block
model.add(Conv2D(70, (3, 3), activation='relu', strides=(2, 2), padding='same'))
model.add(MaxPool2D(pool_size=(2, 2), padding='valid'))
model.add(BatchNormalization())
# ANN block
model.add(Flatten())
model.add(Dense(units=100, activation='relu'))
model.add(Dense(units=100, activation='relu'))
model.add(Dropout(0.25))
# output layer
model.add(Dense(units=10, activation='softmax'))

# compile model
model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])
# fit on data for 30 epochs
model.fit_generator(train, epochs=10, validation_data=val)

"""4. Using Transfer Learning (VGG16) to improve accuracy
VGG16 is a CNN architecture that was the first runner-up in the 2014 ImageNet Challenge. It’s designed by the Visual Graphics Group at Oxford and has 16 layers in total, with 13 convolutional layers themselves. We will load the pre-trained weights of this model so that we can utilize the useful features this model has learned for our task.

Downloading weights of VGG16
"""

from keras.applications import VGG16

# include top should be False to remove the softmax layer
pretrained_model = VGG16(include_top=False, weights='imagenet')
pretrained_model.summary()

"""Generate features from VGG16
Let’s extract useful features that VGG16 already knows from our dataset’s images:
"""

from keras.utils import to_categorical
# extract train and val features
vgg_features_train = pretrained_model.predict(train)
vgg_features_val = pretrained_model.predict(val)

# OHE target column
train_target = to_categorical(train.labels)
val_target = to_categorical(val.labels)

"""Once the above features are ready, we can just use them to train a basic Fully Connected Neural Network in Keras:"""

model2 = Sequential()
model2.add(Flatten(input_shape=(7,7,512)))
model2.add(Dense(100, activation='relu'))
model2.add(Dropout(0.5))
model2.add(BatchNormalization())
model2.add(Dense(10, activation='softmax'))

# compile the model
model2.compile(optimizer='adam', metrics=['accuracy'], loss='categorical_crossentropy')

model2.summary()

# train model using features generated from VGG16 model
model2.fit(vgg_features_train, train_target, epochs=50, batch_size=128, validation_data=(vgg_features_val, val_target))