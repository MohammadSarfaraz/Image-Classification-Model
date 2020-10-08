# Image-Classification-Model
Image Classification with deep Learning using Tensorflow and Keras

# Introduction
* Convolutional neural networks (CNN) – the concept behind recent breakthroughs  and developments in deep learning.

* CNNs have broken the mold and ascended the throne to become the state-of-the-art computer vision technique. Among the different types of neural networks (others include recurrent neural networks (RNN), long short term memory (LSTM), artificial neural networks (ANN), etc.), CNNs are easily the most popular.

There are various datasets that you can leverage for applying convolutional neural networks. Here are three popular datasets:

* MNIST
* CIFAR-10
* ImageNet

# 1.Using CNNs to Classify Hand-written Digits on MNIST Dataset
###  MNIST CNN
![](https://cdn.analyticsvidhya.com/wp-content/uploads/2020/02/mnist.png)

---



* MNIST (Modified National Institute of Standards and Technology) is a well-known dataset used in Computer Vision that was built by Yann Le Cun et. al. It is composed of images that are handwritten digits (0-9), split into a training set of 50,000 images and a test set of 10,000 where each image is of 28 x 28 pixels in width and height.

# 2.Identifying Images from the CIFAR-10 Dataset using CNNs
![](https://cdn.analyticsvidhya.com/wp-content/uploads/2020/02/1_sGochNLZ-qfesdyjadgXNw.png)

The important points that distinguish this dataset from MNIST are:



Images are colored in CIFAR-10 as compared to the black and white texture of MNIST

* Each image is 32 x 32 colour pixel/images in 10 classes,
* 50,000 training images and 
* 10,000 testing images.

Now, these images are taken in varying lighting conditions and at different angles, and since these are colored images, you will see that there are many variations in the color itself of similar objects (for example, the color of ocean water). If you use the simple CNN architecture that we saw in the MNIST example above, you will get a low validation accuracy of around 60%.



Here’s what I had to changed in the model:
* Increased the number of Conv2D layers to build a deeper model
* Increased number of filters to learn more features
* Added Dropout for regularization
* Added more Dense layers

Training and validation accuracy across epochs:

# 3.Categorizing the Images of ImageNet using CNNs

![](https://cdn.analyticsvidhya.com/wp-content/uploads/2020/02/ImageNet-Title-Pic.jpg)

ImageNet is the main database behind the ImageNet Large Scale Recognition Challenge (ILSVRC). This is like the Olympics of Computer Vision. This is the competition that made CNNs popular the first time and every year, the best research teams across industries and academia compete with their best algorithms on computer vision tasks.

 

# About the ImageNet Dataset
The ImageNet dataset has more than 14 million images, hand-labeled across 20,000 categories.

Also, unlike the MNIST and CIFAR-10 datasets that we have already discussed, the images in ImageNet are of decent resolution (224 x 224) and that’s what poses a challenge for us: 14 million images, each 224 by 224 pixels. Processing a dataset of this size requires a great amount of computing power in terms of CPU, GPU, and RAM.

The downside – that might be too much for an everyday laptop. So what’s the alternative solution? How can an enthusiast work with the ImageNet dataset?

 

That’s where Fast.ai’s Imagenette dataset comes in
Imagenette is a dataset that’s extracted from the large ImageNet collection of images. The reason behind releasing Imagenette is that researchers and students can practice on ImageNet level images without needing that much compute resources.

# Loading Images using ImageDataGenerator
Keras has this useful functionality for loading large images (like we have here) without maxing out the RAM, by doing it in small batches. 
ImageDataGenerator in combination with fit_generator provides this functionality:

* Building a Basic CNN model for Image Classification


* Using Transfer Learning (VGG16) to improve accuracy
VGG16 is a CNN architecture that was the first runner-up in the 2014 ImageNet Challenge. It’s designed by the Visual Graphics Group at Oxford and has 16 layers in total, with 13 convolutional layers themselves. We will load the pre-trained weights of this model so that we can utilize the useful features this model has learned for our task.


