"""
Part 1: Simple baseline that only uses word statistics to predict tags
"""

def baseline(train, test):
    '''
    input:  training data (list of sentences, with tags on the words)
            test data (list of sentences, no tags on the words)
    output: list of sentences, each sentence is a list of (word,tag) pairs.
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''
    stopword = [] # List of Stop Words
    # Training Set
    # training_tags[word] = (set(tags), count[tag])

    training_tags = {}
    training_all_tags = {}
    for sentence in train:
        for word, tag in sentence:
            if word in stopword:
                continue
            if word not in training_tags:
                training_tags[word] = {}
            if tag not in training_tags[word]:
                training_tags[word][tag] = 0
            if tag not in training_all_tags:
                training_all_tags[tag] = 0
            training_tags[word][tag] += 1
            training_all_tags[tag] += 1
    
    # Development Set
    default_tag = ""
    max_count = -1
    for tag in training_all_tags:
        if max_count < training_all_tags[tag]:
            max_count = training_all_tags[tag]
            default_tag = tag

    predicted_sentences = []
    for sentence in test:
        output_sentence = [] # [(word1, tag1), (word2, tag2)]
        for word in sentence:
            if word in stopword:
                continue
            if word not in training_tags:
                output_sentence.append((word, default_tag))
                continue
            output_sentence.append((word, find_most_common_tag(training_tags[word], default_tag)))
        predicted_sentences.append(output_sentence)
                
    return predicted_sentences

def find_most_common_tag(training_tags_word, default_tag):
    max_count = 0
    max_tag = default_tag
    if len(training_tags_word) == 0:
        return default_tag
    for tag in training_tags_word:
        count = training_tags_word[tag]
        if max_count < count:
            max_count = count
            max_tag = tag
    return max_tag