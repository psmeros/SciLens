import pickle
import re
from math import sqrt

import nltk.data
import numpy as np
import pandas as pd
import spacy
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.neighbors import NearestNeighbors

from glove import cos_sim, sent2vec
from settings import *
from topic_detection import predict_topic

nlp, tokenizer = (None,)*2

#Classifier to compute feature importance of similarities
def compute_similarity_model(pairs_file, model_out_file=None, cross_val=True):
    fold = 10
    n_est = 80
    m_dep = 80

    df = pd.read_csv(pairs_file, sep='\t')

    #clean some false positives
    df = df[~((df['related']==True) & (df['vec_sim']+df['jac_sim']+df['top_sim']<0.9))]

    #cross validation
    X = df[['vec_sim', 'jac_sim', 'len_sim', 'top_sim']].values
    y = df[['related']].values.ravel()
    
    if cross_val:
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
    else:
        classifier = RandomForestClassifier(n_estimators=n_est, max_depth=m_dep)
        classifier.fit(X, y)
        pickle.dump(classifier, open(model_out_file, 'wb'))


#Extract entities from text
def entity_extraction(text, granularity):

    global nlp, tokenizer
    if nlp is None:
        nlp = spacy.load('en')
        tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    

    if granularity == 'full_text':
        passages = [text] 
    elif granularity == 'paragraph':
        passages = [p for p in re.split('\n', text) if len(p) > MIN_PAR_LENGTH]
    elif granularity == 'sentence':
        passages = [s for p in re.split('\n', text) if len(p) > MIN_PAR_LENGTH for s in tokenizer.tokenize(p) if len(s) > MIN_SEN_LENGTH]

    p_entities = []
    for p in passages:
        entities = []
        
        for v in hn_vocabulary:
            if v in text:
                entities.append(v)
                
        for e in nlp(text).ents:
            if e.text not in entities and e.label_ in ['PERSON', 'ORG', 'DATE', 'TIME', 'PERCENT', 'QUANTITY', 'CARDINAL']:
                entities.append(e.text)
        p_entities.append(set(entities))        
    
    return p_entities[0] if granularity == 'full_text' else p_entities


#Compute the cartesian similarity between the paragraphs of a pair of articles
def cartesian_similarity(pair, granularity='full_text'):
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

    def similarity_features(text, MIN_TEXT_LENGTH=0):
        
        if text not in cache.keys():
            len_t = len(text)
            if (len_t < MIN_TEXT_LENGTH):
                return None

            vec_t = sent2vec(text)
            entities_t = entity_extraction(text)
            topics_t = predict_topic(text)

            cache[text] = (len_t, vec_t, entities_t, topics_t)
        return cache[text]

    if granularity == 'full_text':

        len_x, vec_x, entities_x, topics_x = similarity_features(pair['full_text_x'])
        len_y, vec_y, entities_y, topics_y = similarity_features(pair['full_text_y'])

        vs = vector_similarity(vec_x, vec_y)
        js = jaccard_similarity(entities_x, entities_y)
        ls = len_similarity(len_x, len_y)
        ts = topic_similarity(topics_x, topics_y)

        similarity = [vs, js, ls, ts]

    elif granularity == 'paragraph':
        paragraphs, vs, js, ls, ts = (0, ) * 5
        for par_x in re.split('\n', pair['full_text_x']):
            feat = similarity_features(par_x, MIN_PAR_LENGTH)
            if feat == None:
                continue
            else:
                len_x, vec_x, entities_x, topics_x = feat

            for par_y in re.split('\n', pair['full_text_y']):
                feat = similarity_features(par_y, MIN_PAR_LENGTH)
                if feat == None:
                    continue
                else:
                    len_y, vec_y, entities_y, topics_y = feat
                
                vs += vector_similarity(vec_x, vec_y)
                js += jaccard_similarity(entities_x, entities_y)
                ls += len_similarity(len_x, len_y)
                ts += topic_similarity(topics_x, topics_y)

                paragraphs += 1

        if paragraphs != 0:
            similarity = [vs/paragraphs, js/paragraphs, ls/paragraphs, ts/paragraphs]
        else:
            similarity = [0, 0, 0, 0]

    return similarity

#Compute the similarity for each pair
def compute_pairs_similarity(pairs_file, article_details_file, paper_details_file, granularity, out_file):
    pairs = pd.read_csv(pairs_file, sep='\t')
    article_details = pd.read_csv(article_details_file, sep='\t')
    paper_details = pd.read_csv(paper_details_file, sep='\t')
    print('entities x')
    article_details['entities'] = article_details['full_text'].apply(lambda x: entity_extraction(x, granularity))
    print('topics x')
    article_details['topics'] = article_details['full_text'].apply(lambda x: predict_topic(x, granularity))

    print('entities y')
    paper_details['entities'] = paper_details['full_text'].apply(lambda x: entity_extraction(x, granularity))
    print('topics y')
    paper_details['topics'] = paper_details['full_text'].apply(lambda x: predict_topic(x, granularity))


    pairs = pairs.merge(paper_details[['url', 'full_text', 'entities', 'topics']], left_on='paper', right_on='url')
    pairs = pairs.merge(article_details[['url', 'full_text', 'entities', 'topics']], left_on='article', right_on='url')
    pairs = pairs[['paper', 'full_text_x', 'entities_x', 'topics_x', 'article', 'full_text_y', 'entities_y', 'topics_y', 'related']]
    print(pairs['entities_x'])

    #df = pd.DataFrame(pairs.apply(lambda x: cartesian_similarity(x, granularity=granularity), axis=1).tolist(), columns=['vec_sim', 'jac_sim', 'len_sim', 'top_sim']).reset_index(drop=True)
    #pairs = pairs[['article', 'paper', 'related']].reset_index(drop=True)
    #pairs = pd.concat([pairs, df], axis=1)    
    #pairs.to_csv(out_file, sep='\t', index=None)

#Subsumpling
def uniformly_random_subsample(pairs_file, n_samples, out_file):
    
    pairs = pd.read_csv(pairs_file, sep='\t')
    samples = np.random.uniform(size=(n_samples,pairs.shape[1]-2))

    nn = NearestNeighbors(1, n_jobs=-1)
    nn.fit(pairs[['vec_sim', 'jac_sim', 'len_sim', 'top_sim']])

    index = pd.DataFrame(nn.kneighbors(samples, return_distance=False), columns=['index'])
    df = pairs.reset_index().merge(index).drop_duplicates()

    df.to_csv(out_file, sep='\t', index=None)
