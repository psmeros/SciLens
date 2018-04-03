import spacy
from sklearn.metrics.pairwise import cosine_similarity

from settings import *

nlp = spacy.load('en')

#GloVe Embeddings file
gloveFile = None

gloveEmbeddings = gloveEmbeddingsSize = None

if not os.path.exists(gloveFile):
    print(gloveFile,'embeddings not found!')
    sys.exit(0)

#Load GloVe file
def loadGloveEmbeddings():    
    words = {}
    with open(gloveFile, "r") as f:
        for line in f:
            tokens = line.strip().split()
            words[tokens[0]] = np.array([float(x) for x in tokens[1:]])
  
    global gloveEmbeddings, gloveEmbeddingsSize
    gloveEmbeddings, gloveEmbeddingsSize = words, len(next(iter(words.values())))

#Return the vector of a word
def word2vec(word):
    global gloveEmbeddings, gloveEmbeddingsSize        
    if gloveEmbeddingsSize == None: loadGloveEmbeddings()
    
    try:
        return gloveEmbeddings[word.lower().strip()]
    except:
        return np.zeros(gloveEmbeddingsSize)

    
#Return the vector of a sentence
def sent2Vec(sentence):    
    vec = word2vec('')
    for w in nlp(sentence):
         if not (w.is_stop or w.is_punct):
            vec += word2vec(w.text)
    return vec   

#Return a dictionary of vectors for the given topics
def topics2Vec(topics):
    tEmbeddings = {}
    for t in topics:
        tEmbeddings[t] = sent2Vec(' '.join(t.split(',')))
    return tEmbeddings


#Compute the cosine similarity between two vectors
def sim(vec1, vec2):
    return abs(cosine_similarity(vec1.reshape(1, -1), vec2.reshape(1, -1))[0][0])