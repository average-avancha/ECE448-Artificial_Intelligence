# classify.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 10/27/2018
# Extended by Daniel Gonzales (dsgonza2@illinois.edu) on 3/11/2020

"""
This is the main entry point for MP5. You should only modify code
within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.

train_set - A Numpy array of 32x32x3 images of shape [7500, 3072].
            This can be thought of as a list of 7500 vectors that are each
            3072 dimensional.  We have 3072 dimensions because there are
            each image is 32x32 and we have 3 color channels.
            So 32*32*3 = 3072. RGB values have been scaled to range 0-1.

train_labels - List of labels corresponding with images in train_set
example: Suppose I had two images [X1,X2] where X1 and X2 are 3072 dimensional vectors
         and X1 is a picture of a dog and X2 is a picture of an airplane.
         Then train_labels := [1,0] because X1 contains a picture of an animal
         and X2 contains no animals in the picture.

dev_set - A Numpy array of 32x32x3 images of shape [2500, 3072].
          It is the same format as train_set

return - a list containing predicted labels for dev_set
"""

import numpy as np
import heapq

def trainPerceptron(train_set, train_labels, learning_rate, max_iter):
    # return the trained weight and bias parameters
    """
    For each image in each epoch, if the predicted label does not match the actual label,
    adjust the weights and bias by the learning rate and the corresponding feature vector (multiply the weight in place by the feature matrix)
    in the direction of the actual label (add or subtract from the current weight)
    """
    W = [0] * len(train_set[0])  # Initialize weight matrix to 0
    b = 0 # Initialize bias to 0
    
    for epoch in range(max_iter):
        for image_idx, image in enumerate(train_set):
            predicted = np.sign(np.dot(W, image) + b)
            actual = train_labels[image_idx]
            if actual == 0:
                actual = -1
            if actual != predicted:
                W += learning_rate * actual * image 
                b += learning_rate * actual # Bias has no feature vector to scale by
    return W, b

def classifyPerceptron(train_set, train_labels, dev_set, learning_rate, max_iter):
    # Train perceptron model and return predicted labels of development set
    """
    Train the weights and biases using the training data.
    Predict the labels of an image as list using the tranined weights and bias.
    """
    W, b = trainPerceptron(train_set, train_labels, learning_rate, max_iter)
    
    predicted_labels = []
    for image in dev_set:
        if np.sign(np.dot(W, image) + b) > 0:
            predicted_labels.append(1)
        else:
            predicted_labels.append(0)
    return predicted_labels

def classifyKNN(train_set, train_labels, dev_set, k):
    predicted_labels = []
    for test_image in dev_set:
        distance_list = []
        heapq.heapify(distance_list)
        for training_idx, training_image in enumerate(train_set):
            heapq.heappush(distance_list, (np.sum((test_image - training_image)**2), train_labels[training_idx]))
        label_sum = 0
        for i in range(k):
            distance, label = heapq.heappop(distance_list)
            label_sum += label
        if label_sum/k > 0.5:
            predicted_labels.append(1)
        else:
            predicted_labels.append(0)
    return predicted_labels
