import re
from math import sqrt

import numpy as np
import pandas as pd
import spacy
from sklearn.neighbors import NearestNeighbors

from glove import cos_sim, sent2vec
from settings import *

def text_to_bag_of_entities(text, nlp, vocabulary):
    paragraphs = re.split('\n', text)
    
    text_repr = []
    for p in paragraphs:
        entities = []
        
        for v in vocabulary:
            if v in p:
                entities.append(v)
                
        for e in nlp(p).ents:
            if e.text not in entities and e.label_ in ['PERSON', 'ORG', 'DATE', 'TIME', 'PERCENT', 'QUANTITY', 'CARDINAL']:
                entities.append(e.text)
        
        text_repr.append(entities)
    
    return text_repr

def prepare_articles_matching(in_file, out_file):
    nlp = spacy.load('en')
    vocabulary = open(topicsFile).read().splitlines()

    df = pd.read_csv(in_file, sep='\t')
    df = df[~df['full_text'].isnull()]
    df['entities'] = df['full_text'].apply(lambda x: text_to_bag_of_entities(x, nlp, vocabulary))
    df.to_csv(out_file, sep='\t', index=None)


def vector_similarity(vector_x, vector_y):
    return cos_sim(vector_x, vector_y)

def jaccard_similarity(set_x, set_y):
    return len(set_x.intersection(set_y)) / (len(set_x.union(set_y)) + 0.1)

def len_similarity(len_x, len_y):
    return min(len_x, len_y) / (max(len_x, len_y) + 0.1)

def topic_similarity(tvx, tvy):
    sim = 0
    for tx, ty in zip(tvx, tvy):
        sim += (sqrt(tx) - sqrt(ty))**2
    sim = 1 - 1/sqrt(2) * sqrt(sim)
    return sim

def cartesian_similarity(pair):
    MIN_PAR_LENGTH = 256
    entities_x = eval(pair['entities_x'])
    entities_y = eval(pair['entities_y'])
    full_text_x = re.split('\n', pair['full_text_x'])
    full_text_y = re.split('\n', pair['full_text_y'])
    topics_x = eval(pair['topics_x'])
    topics_y = eval(pair['topics_y'])
    
    similarities = []
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
            
            vs = vector_similarity(ftvx, ftvy)
            js = jaccard_similarity(ex, ey)
            ls = len_similarity(len_x, len_y)
            ts = topic_similarity(tvx, tvy)
            
            similarities.append([ftx, fty, vs, js, ls, ts])
            
    return similarities

def create_annotation_subsample(pairs_file, article_details_file, paper_details_file, subsample_size, out_file):
    pairs = pd.read_csv(pairs_file, sep='\t')
    article_details = pd.read_csv(article_details_file, sep='\t')
    paper_details = pd.read_csv(paper_details_file, sep='\t')

    pairs = pairs.merge(paper_details[['url','entities', 'full_text', 'topics']], left_on='paper', right_on='url')
    pairs = pairs.merge(article_details[['url','entities', 'full_text', 'topics']], left_on='article', right_on='url')
    pairs = pairs[['paper', 'entities_x', 'full_text_x', 'topics_x', 'article', 'entities_y', 'full_text_y', 'topics_y']]

    sample = pairs.sample(subsample_size)
    sample = sample.apply(cartesian_similarity, axis=1)
    df = pd.DataFrame([p for s in sample.tolist() for p in s], columns=['par_x', 'par_y', 'vec_sim', 'jac_sim', 'len_sim', 'top_sim'])
    df.to_csv(out_file, sep='\t', index=None)


def uniformly_random_subsample(pairs_file, n_samples, out_file):
    
    pairs = pd.read_csv('cache/par_pairs_v1.tsv', sep='\t')
    samples = np.random.uniform(size=(n_samples,pairs.shape[1]-2))

    nn = NearestNeighbors(1, n_jobs=-1)
    nn.fit(pairs[['vec_sim', 'jac_sim', 'len_sim', 'top_sim']])

    index = pd.DataFrame(nn.kneighbors(samples, return_distance=False), columns=['index'])
    df = pairs.reset_index().merge(index).drop_duplicates()

    df.to_csv(out_file, sep='\t', index=None)
