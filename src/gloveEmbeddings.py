import os
import numpy as np
from spacy.en import English
from sklearn.metrics.pairwise import cosine_similarity

from settings import *
gloveEmbeddings = gloveEmbeddingsSize = None

#Loads GloVe file.
def loadGloveEmbeddings():    
    words = {}
    with open(gloveFile, "r") as f:
        for line in f:
            tokens = line.strip().split()
            words[tokens[0]] = np.array([float(x) for x in tokens[1:]])
  
    global gloveEmbeddings, gloveEmbeddingsSize
    gloveEmbeddings, gloveEmbeddingsSize = words, len(next(iter(words.values())))

#Returns the vector of a word.
def word2vec(word):
    global gloveEmbeddings, gloveEmbeddingsSize        
    if gloveEmbeddingsSize == None: loadGloveEmbeddings()
    
    try:
        return gloveEmbeddings[word.lower().strip()]
    except:
        return np.zeros(gloveEmbeddingsSize)

    
#Returns the vector of a sentence.
def sent2Vec(sentence):    
    vec = word2vec('')
    for w in nlp(sentence):
         if not (w.is_stop or w.is_punct):
            vec += word2vec(w.text)
    return vec   

#Returns a dictionary of vectors for the given topics.
def topics2Vec(topics):
    tEmbeddings = {}
    for t in topics:
        tEmbeddings[t] = sent2Vec(' '.join(t.split(',')))
    return tEmbeddings


#Computes the cosine similarity between two vectors.
def sim(vec1, vec2):
    return abs(cosine_similarity(vec1.reshape(1, -1), vec2.reshape(1, -1))[0][0])


#Returns the vector of the body of an article.
def body2Vec(body):    
    sentences = [sent.text.strip() for sent in nlp(body).sents] #TODO: change to sent tokenizer
    sentencesVector=[]
    for s in sentences:
        sentencesVector.append(sent2Vec(s))
    return sentencesVector

#Returns the vector of the title of an article.
def title2Vec(title):
    return sent2Vec(title)