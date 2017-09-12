import os
import numpy as np
from nltk import sent_tokenize, wordpunct_tokenize
from nltk.corpus import stopwords
stopWords = stopwords.words("english")


gloveEmbeddings = gloveEmbeddingsSize = None

#Loads the file.
def loadGloveEmbeddings(filename):
    if not os.path.exists(filename):
        print(filename,'embeddings not found')
        return None
    words = {} #key= word, value=embeddings
    with open(filename, "r") as f:
        for line in f:
            tokens = line.strip().split()
            words[tokens[0]] = np.array([float(x) for x in tokens[1:]])
    
    global gloveEmbeddings, gloveEmbeddingsSize
    gloveEmbeddings, gloveEmbeddingsSize = words, len(next(iter(words.values())))

    
#Returns the vector of a word.
def word2vec(word):
    global gloveEmbeddings, gloveEmbeddingsSize
    word = word.lower().strip()
    try:
        return(gloveEmbeddings[word].reshape(1, -1))
    except:
        return np.zeros(gloveEmbeddingsSize).reshape(1, -1)

    
#Returns the vectors of each sentence of the body of an article.
def body2Vec(body):
    sentences = sent_tokenize(body)

    sentencesVector=[]
    for s in sentences:
        vec = word2vec('')
        for w in wordpunct_tokenize(s):
            if len(w)!=1 and w not in stopWords:
                vec += word2vec(w)
        sentencesVector.append(vec)
    return sentencesVector


#Returns the vector of the title of an article.
def title2Vec(title):
    vec = word2vec('')
    for w in wordpunct_tokenize(title):
         if len(w)!=1 and w not in stopWords:
            vec += word2vec(w)
    return vec       
