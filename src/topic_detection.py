import os
import pickle
import re
import sys
from time import time

import nltk.data
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

from settings import *
from utils import split_text_to_passages

lda, tf_vectorizer = (None,)*2

#Discover topics for both articles and quotes
def train_topic_model(news_art_file, sci_art_file, model_file):

    df = pd.concat([pd.read_csv(news_art_file, sep='\t')[['full_text']], pd.read_csv(sci_art_file, sep='\t')[['full_text']]], ignore_index=True)
    
    #define vectorizer (1-2grams)
    tf_vectorizer = CountVectorizer(stop_words='english', vocabulary=hn_vocabulary)
    tf = tf_vectorizer.transform(df['full_text'])

    #fit lda topic model
    print('Fitting LDA model...')
    lda = LatentDirichletAllocation(n_components=numOfTopics, max_iter=max_iter, learning_method='online', n_jobs=-1)
    lda.fit(tf)

    pickle.dump(lda, open(model_file + '.lda', 'wb'))
    pickle.dump(tf_vectorizer, open(model_file + '.vec', 'wb'))

#Predict topic for given text
def predict_topic(text, model_file=cache_dir + 'topic_model'):
    global lda, tf_vectorizer
    if lda is None:
        lda = pickle.load(open(model_file + '.lda', 'rb'))
        tf_vectorizer = pickle.load(open(model_file + '.vec', 'rb'))

    return lda.transform(tf_vectorizer.transform(text)).tolist() if text else []