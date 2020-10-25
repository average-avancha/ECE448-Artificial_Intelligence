"""
Part 2: This is the simplest version of viterbi that doesn't do anything special for unseen words
but it should do better than the baseline at words with multiple tags (because now you're using context
to predict the tag).
"""
from math import log

def viterbi_1(train, test):
    '''
    input:  training data (list of sentences, with tags on the words)
            test data (list of sentences, no tags on the words)
    output: list of sentences with tags on the words
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''
    tuning_constant = 0.1
    min_prob = -999999999
    
    initial, transition, emission, all_tags, all_words, initial_word_count, transition_word_count, total_word_count = training(train)
    default_tag = max_tag(all_tags)
    
    predicted_sentences = []
    for test_count, sentence in enumerate(test):
        # --> Construct Trellis
        trellis, final_tag = buildTrellis(sentence, all_tags, all_words, min_prob, default_tag, tuning_constant, initial_word_count, initial, transition, emission)
        # --> build predicted sentences w/ (word, predicted_tag) 
        predicted_sentences.append(buildSentence(final_tag, trellis, sentence))
    return predicted_sentences

def training(train):
    """
    Hidden Markov Model:
        P(T|W) ∝ P(W|T) ∗ P(T) 
        = ∏i=[1, n] {P(wi|ti) ∗ P(t1|START) ∗ ∏k=[2, n] {P(tk|tk−1)}}
    """
    
    initial = {}    # {NOUN: #, ADJ: #, ...} --> count of a tag at the start of the sentence
    transition = {} # {NOUN: {NOUN: #, ADJ: #, ...}, ADJ:{NOUN: #, ADJ: #, ...} ...} = # --> count of a tag given a previous tag
    emission = {}   # {NOUN: {word1: #, word2: #, ...}, ADJ: {word1: #, word2: #, ...}, ...} = # --> count of a word given a tag for that word
    all_tags = {}   # {NOUN: #, ADJ: #, ...} --> count of a tag
    all_words = {}  # {word1: #, word2: #, ...} --> count of a word
    
    initial_word_count = 0
    transition_word_count = 0
    total_word_count = 0 # used for emission probability
    
    # Training Data --> Counting Tag Occurences
    for sentence in train:
        count = 1
        prev_tag = ""
        for word, tag in sentence:
            # Initial Probability
            if count == 1:
                if tag not in initial:
                    initial[tag] = 0
                    transition[tag] = {}
                initial[tag] += 1
                initial_word_count += 1
            # Transition Probability
            else:
                if prev_tag not in transition:
                        transition[prev_tag] = {}
                if tag not in transition[prev_tag]:
                    transition[prev_tag][tag] = 0
                transition[prev_tag][tag] += 1
                transition_word_count += 1
            # Emission Probability
            if tag not in emission:
                emission[tag] = {}
            if word not in emission[tag]:
                emission[tag][word] = 0
            if tag not in all_tags:
                all_tags[tag] = 0
            if word not in all_words:
                all_words[word] = 0
            all_words[word] += 1
            all_tags[tag] += 1
            emission[tag][word] += 1
            total_word_count += 1
            prev_tag = tag
            count += 1
    return (initial, transition, emission, all_tags, all_words, initial_word_count, transition_word_count, total_word_count)
def computeInitial(tag, initial, alpha, tag_count, initial_n):
    """
    Laplace Smoothing for Initial probability:
        n = count(START)
        V = unique(t1)
        
        Probability of tag (t1) seen in traning data:
            P(t1 | START) = (count(t1)+α)/(n+α(V+1))
        Probability of unseen tag (t1) given START:
            P(UNK | START) = α/(n+α(V+1))
    """
    n = initial_n
    V = len(tag_count)
    if tag not in initial:
        return log(alpha/(n + alpha*(V + 1)))
    return log((initial[tag] + alpha)/(n + alpha*(V + 1)))
    
    
def computeTransition(prev_tag, tag, transition, alpha, tag_count):
    """
    Laplace Smoothing for Transition probability:
        n = count(tk-1)
        V = unique(tk)
        
        Probability of tag (tk) seen in traning data:
            P(tk | tk−1) = (count(tk)+α)/(n+α(V+1))
        Probability of unseen tag (tk) given previous tag (tk-1):
            P(UNK | tk−1) = α/(n+α(V+1))
    """
    n = tag_count[prev_tag]
    V = len(tag_count)
    if prev_tag not in transition:
        transition[prev_tag] = {}
    if tag not in transition[prev_tag]:
        return log((alpha)/(n + alpha*(V + 1)))
    return log((transition[prev_tag][tag] + alpha)/(n + alpha*(V + 1)))

def computeEmission(tag, word, emission, alpha, tag_count, word_count):
    """
    Laplace Smoothing for Emission probability:
        n = count(T)
        V = unique(W)
        
        Probability of W seen in traning data:
            P(W | T) = (count(W)+α)/(n+α(V+1))
        Probability of unseen W given tag (T):
            P(UNK | T) = α/(n+α(V+1))
    """
    n = tag_count[tag]
    V = len(word_count)
    if tag not in emission:
        emission[tag] = {}
    if word not in emission[tag]:
        return log(alpha/(n + alpha*(V + 1)))
    return log((emission[tag][word] + alpha)/(n + alpha*(V + 1)))

def buildTrellis(sentence, all_tags, all_words, min_prob, default_tag, tuning_constant, initial_word_count, initial, transition, emission):
    """
    Hidden Markov Model:
        P(T∣W) ∝ P(W∣T) ∗ P(T) 
        = ∏i=[1, n] {P(wi∣ti) ∗ P(t1∣START) ∗ ∏k=[2, n] {P(tk∣tk−1)}}
    
    Emission Prob. --> P(wi∣ti)
    Initial Prob. --> P(t1∣START)
    Transition Prob. --> P(tk∣tk−1)
    
    Note: Probabilities are stored as log(P) in order to avoid underflow of low probabilities
    """
    trellis = {}
    final_tags = {}
    for i, word in enumerate(sentence):
        trellis[i] = {}
        if i == 0:
            for tag in all_tags:
                p = computeInitial(tag, initial, tuning_constant, all_tags, initial_word_count)
                trellis[i][tag] = (p, (-1, ""))
        else:
            for prev_tag in all_tags:
                for tag in all_tags:
                    transition_p = computeTransition(prev_tag, tag, transition, tuning_constant, all_tags)
                    emission_p = computeEmission(tag, word, emission, tuning_constant, all_tags, all_words)
                    p = trellis[i-1][prev_tag][0] + transition_p + emission_p
                    if tag not in trellis[i]:
                        trellis[i][tag] = (p, (i-1, prev_tag))
                        final_tags[i] = tag
                    if p > trellis[i][tag][0]:
                        trellis[i][tag] = (p, (i-1, prev_tag))
                        final_tags[i] = tag
    return (trellis, final_tags[len(sentence) - 1])

def buildSentence(final_tag, trellis, sentence):
    output_sentence = []
    tag = final_tag
    for i in range(len(sentence) - 1, -1, -1):
        word = sentence[i]
        output_sentence.append((word, tag))
        p, (prev_idx, prev_tag) = trellis[i][tag]
        tag = prev_tag
    output_sentence.reverse()
    return output_sentence

def max_tag(all_tags, default_tag=""):
    max_count = 0
    for tag in all_tags:
        if max_count < all_tags[tag]:
            max_count = all_tags[tag]
            default_tag = tag
    return default_tag