import os
import pickle
import re
import sys
from time import time

import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

from settings import *
from utils import initSpark


#Discover topics for both articles and quotes
def train_model(news_art_file, sci_art_file, model_file):

    df = pd.concat([pd.read_csv(news_art_file, sep='\t')[['full_text']], pd.read_csv(sci_art_file, sep='\t')[['full_text']]], ignore_index=True)

    #use topics as vocabulary to reduce dimensions
    vocabulary = open(topicsFile).read().splitlines()
    
    #define vectorizer (1-2grams)
    tf_vectorizer = CountVectorizer(stop_words='english', vocabulary=vocabulary)
    tf = tf_vectorizer.transform(df['full_text'])

    #fit lda topic model
    print('Fitting LDA model...')
    lda = LatentDirichletAllocation(n_components=numOfTopics, max_iter=max_iter, learning_method='online', n_jobs=-1)
    lda.fit(tf)

    pickle.dump(lda, open(model_file + '.lda', 'wb'))
    pickle.dump(tf_vectorizer, open(model_file + '.vec', 'wb'))

#Predict topics for articles
def predict_topic(news_art_in_file, news_art_out_file, sci_art_in_file, sci_art_out_file, model_file):
    lda = pickle.load(open(model_file + '.lda', 'rb'))
    tf_vectorizer = pickle.load(open(model_file + '.vec', 'rb'))

    print('Discovering news article topics...')
    df = pd.read_csv(news_art_in_file, sep='\t')
    df['topics'] = df['full_text'].apply(lambda x: lda.transform(tf_vectorizer.transform(re.split('\n', x))))
    df.to_csv(news_art_out_file, sep='\t', index=None)

    print('Discovering scientific article topics...')
    df = pd.read_csv(sci_art_in_file, sep='\t')
    df['topics'] = df['full_text'].apply(lambda x: lda.transform(tf_vectorizer.transform(re.split('\n', x))))
    df.to_csv(sci_art_out_file, sep='\t', index=None)
