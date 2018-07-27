import os
import sys
import numpy as np

import spacy
from sklearn.metrics.pairwise import cosine_similarity

from settings import *

nlp = spacy.load('en')


#Load GloVe file
def load_Glove():    
    words = {}
    with open(glove_file, "r") as f:
        for line in f:
            tokens = line.strip().split()
            words[tokens[0]] = np.array([float(x) for x in tokens[1:]])
  
    return words, len(next(iter(words.values())))

glove_embeddings, glove_embeddings_size = load_Glove() 

#Return the vector of a word
def word2vec(word):
    try:
        return glove_embeddings[word.lower().strip()]
    except:
        return np.zeros(glove_embeddings_size)

    
#Return the vector of a sentence
def sent2vec(sentence):    
    vec = word2vec('')
    if len(sentence) != 0:
        for w in nlp(sentence):
            if not (w.is_stop or w.is_punct):
                vec += word2vec(w.text)
    return vec   

#Compute the cosine similarity between two vectors
def cos_sim(vec1, vec2):
    return abs(cosine_similarity(vec1.reshape(1, -1), vec2.reshape(1, -1))[0][0])