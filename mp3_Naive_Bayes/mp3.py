# mp3.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 09/28/2018
import sys
import argparse
import configparser
import copy
import numpy as np

import reader
import naive_bayes as nb
from sklearn.metrics import confusion_matrix

"""
This file contains the main application that is run for this MP.
"""

def compute_accuracies(predicted_labels, dev_labels):
    yhats = predicted_labels
    accuracy = np.mean(yhats == dev_labels)
    cm = confusion_matrix(dev_labels, predicted_labels)
    tn, fp, fn, tp = cm.ravel()
    true_negative = tn
    false_positive = fp
    false_negative = fn
    true_positive = tp
    return accuracy, false_positive, false_negative, true_positive, true_negative


def main(args):
    #Modify stemming and lower case below. Note that our test cases may use both settings of the two parameters
    max_iterations = 10
    accuracy_limit = 0.87
    
    min_accuracy = 0
    max_accuracy = 0
    
    unigram_smoothing_parameter = 0.0625
    bigram_smoothing_parameter = 0.125
    bigram_lambda = 0.05
    # unigram smoothing parameter tuning domain
    min_unigram_smoothing_parameter = 0.0000001
    max_unigram_smoothing_parameter = 1.0
    # bigram smoothing parameter tuning domain
    min_bigram_smoothing_parameter = 0.0000001
    max_bigram_smoothing_parameter = 1.0
    # bigram_lambda tuning domain
    min_bigram_lambda = 0.0000001
    max_bigram_lambda = 1.0
    
    #bigram_lambda tuner    
    iteration = 0
    while min_accuracy < accuracy_limit or max_accuracy < accuracy_limit:
        if iteration > max_iterations:
            break
        # min_bigram_lambda
        train_set, train_labels, dev_set, dev_labels = reader.load_dataset(args.training_dir,args.development_dir,stemming=False,lower_case=False)
        predicted_labels = nb.bigramBayes(train_set, train_labels, dev_set, unigram_smoothing_parameter, bigram_smoothing_parameter, min_bigram_lambda)
        min_accuracy, false_positive, false_negative,true_positive, true_negative = compute_accuracies(predicted_labels,dev_labels)
        # max_bigram_lambda
        train_set, train_labels, dev_set, dev_labels = reader.load_dataset(args.training_dir,args.development_dir,stemming=False,lower_case=False)
        predicted_labels = nb.bigramBayes(train_set, train_labels, dev_set, unigram_smoothing_parameter, bigram_smoothing_parameter, max_bigram_lambda)
        max_accuracy, false_positive, false_negative,true_positive, true_negative = compute_accuracies(predicted_labels,dev_labels)
        
        print("Iteration:", iteration)
        print("unigram_smoothing_parameter:", unigram_smoothing_parameter)
        print("bigram_smoothing_parameter:", bigram_smoothing_parameter)
        print("min_bigram_lambda:",min_bigram_lambda)
        print("max_bigram_lambda:",max_bigram_lambda)
        print("min_Accuracy:", min_accuracy)
        print("max_Accuracy:", max_accuracy)
        print("False Positive:", false_positive)
        print("False Negative:", false_negative)
        print("True Positive:", true_positive)
        print("True Negative:", true_negative)
        
        if(min_accuracy < max_accuracy):
            min_bigram_lambda += (max_bigram_lambda - min_bigram_lambda)/2
            bigram_lambda = max_bigram_lambda
        else:
            max_bigram_lambda -= (max_bigram_lambda - min_bigram_lambda)/2
            bigram_lambda = min_bigram_lambda
        iteration += 1
        
    # unigram_smoothing_parameter tuner
    iteration = 0
    while min_accuracy < accuracy_limit or max_accuracy < accuracy_limit:
        if iteration > max_iterations:
            break
        # min_unigram_smoothing_parameter
        train_set, train_labels, dev_set, dev_labels = reader.load_dataset(args.training_dir,args.development_dir,stemming=False,lower_case=False)
        predicted_labels = nb.bigramBayes(train_set, train_labels, dev_set, min_unigram_smoothing_parameter, bigram_smoothing_parameter, bigram_lambda)
        min_accuracy, false_positive, false_negative,true_positive, true_negative = compute_accuracies(predicted_labels,dev_labels)
        # max_unigram_smoothing_parameter 
        train_set, train_labels, dev_set, dev_labels = reader.load_dataset(args.training_dir,args.development_dir,stemming=False,lower_case=False)
        predicted_labels = nb.bigramBayes(train_set, train_labels, dev_set, max_unigram_smoothing_parameter, bigram_smoothing_parameter, bigram_lambda)
        max_accuracy, false_positive, false_negative,true_positive, true_negative = compute_accuracies(predicted_labels,dev_labels)
        
        print("Iteration:", iteration)
        print("min_unigram_smoothing_parameter:", min_unigram_smoothing_parameter)
        print("max_unigram_smoothing_parameter:", max_unigram_smoothing_parameter)
        print("bigram_smoothing_parameter:", bigram_smoothing_parameter)
        print("bigram_lambda:",bigram_lambda)
        print("min_Accuracy:", min_accuracy)
        print("max_Accuracy:", max_accuracy)
        print("False Positive:", false_positive)
        print("False Negative:", false_negative)
        print("True Positive:", true_positive)
        print("True Negative:", true_negative)
        
        if(min_accuracy < max_accuracy):
            min_unigram_smoothing_parameter += (max_unigram_smoothing_parameter - min_unigram_smoothing_parameter)/2
            unigram_smoothing_parameter = max_unigram_smoothing_parameter
        else:
            max_unigram_smoothing_parameter -= (max_unigram_smoothing_parameter - min_unigram_smoothing_parameter)/2
            unigram_smoothing_parameter = min_unigram_smoothing_parameter
        iteration += 1
    
    # bigram_smoothing_parameter tuner
    iteration = 0
    while min_accuracy < accuracy_limit or max_accuracy < accuracy_limit:
        if iteration > max_iterations:
            break
        # min_bigram_smoothing_parameter
        train_set, train_labels, dev_set, dev_labels = reader.load_dataset(args.training_dir,args.development_dir,stemming=False,lower_case=False)
        predicted_labels = nb.bigramBayes(train_set, train_labels, dev_set, unigram_smoothing_parameter, min_bigram_smoothing_parameter, bigram_lambda)
        min_accuracy, false_positive, false_negative,true_positive, true_negative = compute_accuracies(predicted_labels,dev_labels)
        # max_bigram_smoothing_parameter 
        train_set, train_labels, dev_set, dev_labels = reader.load_dataset(args.training_dir,args.development_dir,stemming=False,lower_case=False)
        predicted_labels = nb.bigramBayes(train_set, train_labels, dev_set, unigram_smoothing_parameter, max_bigram_smoothing_parameter, bigram_lambda)
        max_accuracy, false_positive, false_negative,true_positive, true_negative = compute_accuracies(predicted_labels,dev_labels)
        
        print("Iteration:", iteration)
        print("unigram_smoothing_parameter:", unigram_smoothing_parameter)
        print("min_bigram_smoothing_parameter:", min_bigram_smoothing_parameter)
        print("max_bigram_smoothing_parameter:", max_bigram_smoothing_parameter)
        print("bigram_lambda:",bigram_lambda)
        print("min_Accuracy:", min_accuracy)
        print("max_Accuracy:", max_accuracy)
        print("False Positive:", false_positive)
        print("False Negative:", false_negative)
        print("True Positive:", true_positive)
        print("True Negative:", true_negative)
        
        if(min_accuracy < max_accuracy):
            min_bigram_smoothing_parameter += (max_bigram_smoothing_parameter - min_bigram_smoothing_parameter)/2
            bigram_smoothing_parameter = max_bigram_smoothing_parameter
        else:
            max_bigram_smoothing_parameter -= (max_bigram_smoothing_parameter - min_bigram_smoothing_parameter)/2
            bigram_smoothing_parameter = min_bigram_smoothing_parameter
        iteration += 1
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CS440 MP3 Naive Bayes')
    parser.add_argument('--training', dest='training_dir', type=str, default = '../data/movies_review/train',
                        help='the directory of the training data')
    parser.add_argument('--development', dest='development_dir', type=str, default = '../data/movies_review/dev',
                        help='the directory of the development data')
    args = parser.parse_args()
    main(args)
