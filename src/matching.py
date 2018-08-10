import re
from math import sqrt

import numpy as np
import pandas as pd
import spacy
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.neighbors import NearestNeighbors

from glove import cos_sim, sent2vec
from settings import *

nlp = None

#Classifier to compute feature importance of similarities
def compute_similarity_model(pairs_file):
    fold = 10
    n_est = 80
    m_dep = 80

    df = pd.read_csv(pairs_file, sep='\t')

    #clean some false positives
    df = df[~((df['related']==True) & (df['vec_sim']+df['jac_sim']+df['top_sim']<0.9))]

    #cross validation
    X = df[['vec_sim', 'jac_sim', 'len_sim', 'top_sim']].values
    y = df[['related']].values.ravel()
    kf = KFold(n_splits=fold, shuffle=True)
    score = 0.0
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        classifier = RandomForestClassifier(n_estimators=n_est, max_depth=m_dep)
        classifier.fit(X_train, y_train)
        score += classifier.score(X_test, y_test)

    print('Score:', score/fold)
    print('Feature Importances:', classifier.feature_importances_)


#Extract entities from text
def prepare_articles_matching(in_file, out_file):

    def f(text):
        global nlp
        if nlp is None:
            nlp = spacy.load('en')

        paragraphs = re.split('\n', text)
        
        text_repr = []
        for p in paragraphs:
            entities = []
            
            for v in hn_vocabulary:
                if v in p:
                    entities.append(v)
                    
            for e in nlp(p).ents:
                if e.text not in entities and e.label_ in ['PERSON', 'ORG', 'DATE', 'TIME', 'PERCENT', 'QUANTITY', 'CARDINAL']:
                    entities.append(e.text)
            
            text_repr.append(entities)
        
        return text_repr

    df = pd.read_csv(in_file, sep='\t')
    df = df[~df['full_text'].isnull()]
    df['entities'] = df['full_text'].apply(lambda x: f(x))
    df.to_csv(out_file, sep='\t', index=None)


#Compute the cartesian similarity between the paragraphs of a pair of articles
def cartesian_similarity(pair):
    def vector_similarity(vector_x, vector_y):
        return cos_sim(vector_x, vector_y)

    def jaccard_similarity(set_x, set_y):
        return len(set_x.intersection(set_y)) / (len(set_x.union(set_y)) + 0.1)

    def len_similarity(len_x, len_y):
        return min(len_x, len_y) / (max(len_x, len_y) + 0.1)

    def topic_similarity(tvx, tvy): 
        sim = 0.0
        #if both are not related to any topic return 0
        if not (len(set(tvx)) == len(set(tvy)) == 1 and set(tvx).pop() == set(tvy).pop() == 0.0625):    
            for tx, ty in zip(tvx, tvy):
                sim += (sqrt(tx) - sqrt(ty))**2
            sim = 1 - 1/sqrt(2) * sqrt(sim)
        return sim

    MIN_PAR_LENGTH = 256
    entities_x = eval(pair['entities_x'])
    entities_y = eval(pair['entities_y'])
    full_text_x = re.split('\n', pair['full_text_x'])
    full_text_y = re.split('\n', pair['full_text_y'])
    topics_x = eval(pair['topics_x'])
    topics_y = eval(pair['topics_y'])
    
    paragraphs, vs, js, ls, ts = (0, ) * 5
    for ex, ftx, tvx in zip(entities_x, full_text_x, topics_x):
        len_x = len(ftx)
        if (len_x < MIN_PAR_LENGTH):
            continue
        ex = set(ex)
        ftvx = sent2vec(ftx)
        for ey, fty, tvy in zip(entities_y, full_text_y, topics_y):
            len_y = len(fty)
            if (len_y < MIN_PAR_LENGTH):
                continue    
            ey = set(ey)
            ftvy = sent2vec(fty)
            
            vs += vector_similarity(ftvx, ftvy)
            js += jaccard_similarity(ex, ey)
            ls += len_similarity(len_x, len_y)
            ts += topic_similarity(tvx, tvy)
            paragraphs += 1

    if paragraphs != 0:
        similarity = [vs/paragraphs, js/paragraphs, ls/paragraphs, ts/paragraphs]
    else:
        similarity = [0, 0, 0, 0]
    return similarity

#Compute the similarity for each pair
def compute_pairs_similarity(pairs_file, article_details_file, paper_details_file, out_file):
    pairs = pd.read_csv(pairs_file, sep='\t')
    article_details = pd.read_csv(article_details_file, sep='\t')
    paper_details = pd.read_csv(paper_details_file, sep='\t')

    pairs = pairs.merge(paper_details[['url','entities', 'full_text', 'topics']], left_on='paper', right_on='url')
    pairs = pairs.merge(article_details[['url','entities', 'full_text', 'topics']], left_on='article', right_on='url')
    pairs = pairs[['paper', 'entities_x', 'full_text_x', 'topics_x', 'article', 'entities_y', 'full_text_y', 'topics_y', 'related']]

    df = pd.DataFrame(pairs.apply(cartesian_similarity, axis=1).tolist(), columns=['vec_sim', 'jac_sim', 'len_sim', 'top_sim']).reset_index(drop=True)
    pairs = pairs[['article', 'paper', 'related']].reset_index(drop=True)
    pairs = pd.concat([pairs, df], axis=1)    
    pairs.to_csv(out_file, sep='\t', index=None)

#Subsumpling
def uniformly_random_subsample(pairs_file, n_samples, out_file):
    
    pairs = pd.read_csv(pairs_file, sep='\t')
    samples = np.random.uniform(size=(n_samples,pairs.shape[1]-2))

    nn = NearestNeighbors(1, n_jobs=-1)
    nn.fit(pairs[['vec_sim', 'jac_sim', 'len_sim', 'top_sim']])

    index = pd.DataFrame(nn.kneighbors(samples, return_distance=False), columns=['index'])
    df = pairs.reset_index().merge(index).drop_duplicates()

    df.to_csv(out_file, sep='\t', index=None)
