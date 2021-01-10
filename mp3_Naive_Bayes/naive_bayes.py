# naive_bayes.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 09/28/2018

"""
This is the main entry point for MP3. You should only modify code
within this file and the last two arguments of line 34 in mp3.py
and if you want-- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""

import math

def naiveBayes(train_set, train_labels, dev_set, smoothing_parameter=1.0, pos_prior=0.8):
    """
    train_set - List of list of words corresponding with each movie review
    example: suppose I had two reviews 'like this movie' and 'i fall asleep' in my training set
    Then train_set := [['like','this','movie'], ['i','fall','asleep']]

    train_labels - List of labels corresponding with train_set
    example: Suppose I had two reviews, first one was positive and second one was negative.
    Then train_labels := [1, 0]

    dev_set - List of list of words corresponding with each review that we are testing on
              It follows the same format as train_set

    smoothing_parameter - The smoothing parameter --laplace (1.0 by default)
    pos_prior - The prior probability that a word is positive. You do not need to change this value.
    """
    # TODO: Write your code here
    # return predicted labels of development set
    
    stopwords = ['ourselves', 'hers', 'between', 'yourself', 'but', 'again', 'there', 'about', 'once', 'during', 'out', 'very', 'having', 'with', 'they', 'own', 'an', 'be', 'some', 'for', 'do', 'its', 'yours', 'such', 'into', 'of', 'most', 'itself', 'other', 'off', 'is', 's', 'am', 'or', 'who', 'as', 'from', 'him', 'each', 'the', 'themselves', 'until', 'below', 'are', 'we', 'these', 'your', 'his', 'through', 'don', 'nor', 'me', 'were', 'her', 'more', 'himself', 'this', 'down', 'should', 'our', 'their', 'while', 'above', 'both', 'up', 'to', 'ours', 'had', 'she', 'all', 'no', 'when', 'at', 'any', 'before', 'them', 'same', 'and', 'been', 'have', 'in', 'will', 'on', 'does', 'yourselves', 'then', 'that', 'because', 'what', 'over', 'why', 'so', 'can', 'did', 'not', 'now', 'under', 'he', 'you', 'herself', 'has', 'just', 'where', 'too', 'only', 'myself', 'which', 'those', 'i', 'after', 'few', 'whom', 't', 'being', 'if', 'theirs', 'my', 'against', 'a', 'by', 'doing', 'it', 'how', 'further', 'was', 'here', 'than']
    
    WordGivenPos = {}
    WordGivenNeg = {}
    count = {}
    
    # Unigram Training Code
    n_pos = 0
    n_neg = 0
    for review_num in range(len(train_set)):
        review = train_set[review_num]        
        for word in review:
            if word in stopwords:
                continue
            if train_labels[review_num] == 1: # Type = Positive 
                n_pos += 1
                if word not in WordGivenPos:
                    count[word] = 1
                    WordGivenPos[word] = 1
                else:
                    count[word] += 1
                    WordGivenPos[word] += 1
            else: # train_labels[review_num] == 0 # Type = Negative
                n_neg += 1
                if word not in WordGivenNeg:
                    count[word] = 1
                    WordGivenNeg[word] = 1
                else:
                    count[word] += 1
                    WordGivenNeg[word] += 1
    
    # Unigram Development Code
    predicted_labels = []
    for review in dev_set:
        PosGivenWords = math.log(pos_prior)
        NegGivenWords = math.log(1 - pos_prior)
        for word in review:
            if word in stopwords:
                continue
            if word not in WordGivenPos:
                PosGivenWords = PosGivenWords + math.log(smoothing_parameter/(n_pos + smoothing_parameter*(len(WordGivenPos) + 1)))
            else:
                PosGivenWords = PosGivenWords + math.log((WordGivenPos[word] + smoothing_parameter)/(n_pos + smoothing_parameter*(len(WordGivenPos) + 1)))
            if word not in WordGivenNeg:
                NegGivenWords = NegGivenWords + math.log(smoothing_parameter/(n_neg + smoothing_parameter*(len(WordGivenNeg) + 1)))
            else:
                NegGivenWords = NegGivenWords + math.log((WordGivenNeg[word] + smoothing_parameter)/(n_neg + smoothing_parameter*(len(WordGivenNeg) + 1)))
        if PosGivenWords >= NegGivenWords:
            predicted_labels.append(1)
        else:
            predicted_labels.append(0)
            
    return predicted_labels

def bigramBayes(train_set, train_labels, dev_set, unigram_smoothing_parameter=.125, bigram_smoothing_parameter=.001, bigram_lambda=0.00005,pos_prior=0.8):
    
    # unigram_smoothing_parameter=.03125, bigram_smoothing_parameter=.125, bigram_lambda=0.005,pos_prior=0.8
    
    """
    train_set - List of list of words corresponding with each movie review
    example: suppose I had two reviews 'like this movie' and 'i fall asleep' in my training set
    Then train_set := [['like','this','movie'], ['i','fall','asleep']]

    train_labels - List of labels corresponding with train_set
    example: Suppose I had two reviews, first one was positive and second one was negative.
    Then train_labels := [1, 0]

    dev_set - List of list of words corresponding with each review that we are testing on
              It follows the same format as train_set

    unigram_smoothing_parameter - The smoothing parameter for unigram model (same as above) --laplace (1.0 by default)
    bigram_smoothing_parameter - The smoothing parameter for bigram model (1.0 by default)
    bigram_lambda - Determines what fraction of your prediction is from the bigram model and what fraction is from the unigram model. Default is 0.5
    pos_prior - The prior probability that a word is positive. You do not need to change this value.
    """
    # TODO: Write your code here
    # return predicted labels of development set using a bigram model
    
    stopwords = ['ourselves', 'hers', 'between', 'yourself', 'but', 'again', 'there', 'about', 'once', 'during', 'out', 'very', 'having', 'with', 'they', 'own', 'an', 'be', 'some', 'for', 'do', 'its', 'yours', 'such', 'into', 'of', 'most', 'itself', 'other', 'off', 'is', 's', 'am', 'or', 'who', 'as', 'from', 'him', 'each', 'the', 'themselves', 'until', 'below', 'are', 'we', 'these', 'your', 'his', 'through', 'don', 'nor', 'me', 'were', 'her', 'more', 'himself', 'this', 'down', 'should', 'our', 'their', 'while', 'above', 'both', 'up', 'to', 'ours', 'had', 'she', 'all', 'no', 'when', 'at', 'any', 'before', 'them', 'same', 'and', 'been', 'have', 'in', 'will', 'on', 'does', 'yourselves', 'then', 'that', 'because', 'what', 'over', 'why', 'so', 'can', 'did', 'not', 'now', 'under', 'he', 'you', 'herself', 'has', 'just', 'where', 'too', 'only', 'myself', 'which', 'those', 'i', 'after', 'few', 'whom', 't', 'being', 'if', 'theirs', 'my', 'against', 'a', 'by', 'doing', 'it', 'how', 'further', 'was', 'here', 'than']
    
    
    # Training Code
    n_pos = 0
    n_neg = 0
    bigram_n_pos = 0
    bigram_n_neg = 0
    WordGivenPos = {}
    WordGivenNeg = {}
    count = {}
    bigram_WordGivenPos = {}
    bigram_WordGivenNeg = {}
    bigram_count = {}
    
    for review_num in range(len(train_set)):
        review = train_set[review_num]        
        # Unigram Training
        for word in review:
            if word in stopwords:
                continue
            if train_labels[review_num] == 1: 
                # Type = Positive 
                n_pos += 1
                if word not in WordGivenPos:
                    count[word] = 1
                    WordGivenPos[word] = 1
                else:
                    count[word] += 1
                    WordGivenPos[word] += 1
            else: # train_labels[review_num] == 0 
                # Type = Negative
                n_neg += 1
                if word not in WordGivenNeg:
                    count[word] = 1
                    WordGivenNeg[word] = 1
                else:
                    count[word] += 1
                    WordGivenNeg[word] += 1
        # Bigram Training
        for word_idx in range(len(review) - 1):
            word_pair = review[word_idx] + review[word_idx + 1]
            if (review[word_idx] in stopwords) or (review[word_idx + 1] in stopwords):
                continue
            if train_labels[review_num] == 1:
                # Type = Positive
                bigram_n_pos += 1
                if word_pair not in bigram_WordGivenPos:
                    bigram_count[word_pair] = 1
                    bigram_WordGivenPos[word_pair] = 1
                else:
                    bigram_count[word_pair] += 1
                    bigram_WordGivenPos[word_pair] += 1
            else: # train_labels[review_num] == 0 
                # Type = Negative
                bigram_n_neg += 1
                if word_pair not in bigram_WordGivenNeg:
                    bigram_count[word_pair] = 1
                    bigram_WordGivenNeg[word_pair] = 1
                else:
                    bigram_count[word_pair] += 1
                    bigram_WordGivenNeg[word_pair] += 1
        
    
    # Development Code
    predicted_labels = []
    for review in dev_set:
        PosGivenWords = math.log(pos_prior)
        NegGivenWords = math.log(1 - pos_prior)
        bigram_PosGivenWords = math.log(pos_prior)
        bigram_NegGivenWords = math.log(1 - pos_prior)
        # Unigram Development Code
        for word in review:
            if word in stopwords:
                continue
            if word not in WordGivenPos:
                PosGivenWords = PosGivenWords + math.log(unigram_smoothing_parameter/(n_pos + unigram_smoothing_parameter*(len(WordGivenPos) + 1)))
            else:
                PosGivenWords = PosGivenWords + math.log((WordGivenPos[word] + unigram_smoothing_parameter)/(n_pos + unigram_smoothing_parameter*(len(WordGivenPos) + 1)))
            if word not in WordGivenNeg:
                NegGivenWords = NegGivenWords + math.log(unigram_smoothing_parameter/(n_neg + unigram_smoothing_parameter*(len(WordGivenNeg) + 1)))
            else:
                NegGivenWords = NegGivenWords + math.log((WordGivenNeg[word] + unigram_smoothing_parameter)/(n_neg + unigram_smoothing_parameter*(len(WordGivenNeg) + 1)))
        # Bigram Development Code
        for word_idx in range(len(review) - 1):
            word_pair = review[word_idx] + review[word_idx + 1]
            if (review[word_idx] in stopwords) or (review[word_idx + 1] in stopwords):
                continue
            if word_pair not in bigram_WordGivenPos:
                bigram_PosGivenWords = bigram_PosGivenWords + math.log(bigram_smoothing_parameter/(bigram_n_pos + bigram_smoothing_parameter*(len(bigram_WordGivenPos) + 1)))
            else:
                bigram_PosGivenWords = bigram_PosGivenWords + math.log((bigram_WordGivenPos[word_pair] + bigram_smoothing_parameter)/(bigram_n_pos + bigram_smoothing_parameter*(len(bigram_WordGivenPos) + 1)))
            if word_pair not in bigram_WordGivenNeg:
                bigram_NegGivenWords = bigram_NegGivenWords + math.log(bigram_smoothing_parameter/(bigram_n_neg + bigram_smoothing_parameter*(len(bigram_WordGivenNeg) + 1)))
            else:
                bigram_NegGivenWords = bigram_NegGivenWords + math.log((bigram_WordGivenNeg[word_pair] + bigram_smoothing_parameter)/(bigram_n_neg + bigram_smoothing_parameter*(len(bigram_WordGivenNeg) + 1)))
        
        PosGivenWords = (1 - bigram_lambda) * PosGivenWords + bigram_lambda * bigram_PosGivenWords
        NegGivenWords = (1 - bigram_lambda) * NegGivenWords + bigram_lambda * bigram_NegGivenWords
        if PosGivenWords >= NegGivenWords:
            predicted_labels.append(1)
        else:
            predicted_labels.append(0)
            
    return predicted_labels