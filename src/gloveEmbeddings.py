import os
import numpy as np
from nltk import sent_tokenize, wordpunct_tokenize
from nltk.corpus import stopwords
stopWords = stopwords.words("english")
from sklearn.metrics.pairwise import cosine_similarity

gloveEmbeddings = gloveEmbeddingsSize = None

#Loads the file.
def loadGloveEmbeddings():    
    try: 
        gloveFile = os.environ['gloveFile']
    except: 
        gloveFile = ('/home/psmeros/var/workspaces/nutrition-workspace/bigFiles/glove.6B.50d.txt' if sys.platform == 'linux' else '/Users/smeros/workspace/etc/bigFiles/glove.6B/glove.6B.50d.txt')

    if not os.path.exists(gloveFile):
        print(gloveFile,'embeddings not found')
        return None
    words = {} #key= word, value=embeddings
    with open(gloveFile, "r") as f:
        for line in f:
            tokens = line.strip().split()
            words[tokens[0]] = np.array([float(x) for x in tokens[1:]])
    
    return words, len(next(iter(words.values())))

    
#Returns the vector of a word.
def word2vec(word):
    global gloveEmbeddings, gloveEmbeddingsSize
    
    if gloveEmbeddingsSize == None:
        gloveEmbeddings, gloveEmbeddingsSize = loadGloveEmbeddings(gloveFile)
   
    word = word.lower().strip()
    try:
        return(gloveEmbeddings[word])
    except:
        return np.zeros(gloveEmbeddingsSize)

#Computes the cosine similarity between two vectors.
def sim(vec1, vec2):
    return abs(cosine_similarity(vec1.reshape(1, -1), vec2.reshape(1, -1)))
    
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
    return sent2Vec(title)   


#Returns the vector of a sentence.
def sent2Vec(sentence):
    vec = word2vec('')
    for w in wordpunct_tokenize(sentence):
         if len(w)!=1 and w not in stopWords:
            vec += word2vec(w)
    return vec   
