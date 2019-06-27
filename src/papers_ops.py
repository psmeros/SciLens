from pathlib import Path
import pickle
import re
from math import sqrt
from random import randint

import nltk.data
import numpy as np
import pandas as pd
import spacy
import torch
import torch.nn as nn
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import KFold
from sklearn.neighbors import NearestNeighbors
from sklearn.svm import SVC

from create_corpus import read_graph

nlp = None
tokenizer = None

############################### CONSTANTS ###############################

scilens_dir = str(Path.home()) + '/Dropbox/scilens/'

#Topic Discovery parameters
numOfTopics = 16
max_iter = 100

#Minimum length for articles/paragraphs/sentences (#chars)
MIN_ART_LENGTH = 256
MIN_PAR_LENGTH = 256
MIN_SEN_LENGTH = 32

#GloVe (deprecated)
glove_file = scilens_dir + 'big_files/glove.6B.300d.txt'

#File with refined topics
hn_vocabulary = open(scilens_dir + 'small_files/hn_vocabulary/hn_vocabulary.txt').read().splitlines()


############################### ######### ###############################

################################ HELPERS ################################

#Split text to passages in multiple granularities
def split_text_to_passages(text, granularity):
    global tokenizer
    if tokenizer == None:
        tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

    if granularity == 'full_text':
        passages = [text] if len(text) > MIN_ART_LENGTH else []
    elif granularity == 'paragraph':
        passages = [p for p in re.split('\n', text) if len(p) > MIN_PAR_LENGTH]
    elif granularity == 'sentence':
        passages = [s for p in re.split('\n', text) if len(p) > MIN_PAR_LENGTH for s in tokenizer.tokenize(p) if len(s) > MIN_SEN_LENGTH]
    
    return passages

#Subsumpling
def uniformly_random_subsample(pairs_file, n_samples, out_file):
    
    pairs = pd.read_csv(pairs_file, sep='\t')
    samples = np.random.uniform(size=(n_samples,pairs.shape[1]-2))

    nn = NearestNeighbors(1, n_jobs=-1)
    nn.fit(pairs[['vec_sim', 'jac_sim', 'len_sim', 'top_sim']])

    index = pd.DataFrame(nn.kneighbors(samples, return_distance=False), columns=['index'])
    df = pairs.reset_index().merge(index).drop_duplicates()

    df.to_csv(out_file, sep='\t', index=None)

############################### TOPIC ###############################

lda, tf_vectorizer = (None,)*2

#Discover topics for passages
def train_topic_model(news_art_file, sci_art_file):

    df = pd.concat([pd.read_csv(news_art_file, sep='\t')[['full_text']], pd.read_csv(sci_art_file, sep='\t')[['full_text']]], ignore_index=True)
    
    #define vectorizer (1-2grams)
    tf_vectorizer = CountVectorizer(stop_words='english', vocabulary=hn_vocabulary)
    tf = tf_vectorizer.transform(df['full_text'])

    #fit lda topic model
    print('Fitting LDA model...')
    lda = LatentDirichletAllocation(n_components=numOfTopics, max_iter=max_iter, learning_method='online', n_jobs=-1)
    lda.fit(tf)

    pickle.dump(lda, open(scilens_dir+'/topic_model.lda', 'wb'))
    pickle.dump(tf_vectorizer, open(scilens_dir+'/topic_model.vec', 'wb'))

#Predict topic for given passage
def predict_topic(text):
    global lda, tf_vectorizer
    if lda is None:
        lda = pickle.load(open(scilens_dir+'/topic_model.lda', 'rb'))
        tf_vectorizer = pickle.load(open(scilens_dir+'/topic_model.vec', 'rb'))

    return lda.transform(tf_vectorizer.transform(text)).tolist() if text else []


############################### GLOVE ###############################

nlp, glove_embeddings, glove_embeddings_size  = (None,)*3

#Load GloVe file
def load_Glove():    
    words = {}
    with open(glove_file, "r") as f:
        for line in f:
            tokens = line.strip().split()
            words[tokens[0]] = np.array([float(x) for x in tokens[1:]])
  
    return words, len(next(iter(words.values())))


#Return the vector of a word
def word2vec(word):
    global glove_embeddings, glove_embeddings_size
    if glove_embeddings is None:
        glove_embeddings, glove_embeddings_size = load_Glove()

    try:
        return glove_embeddings[word.lower().strip()]
    except:
        return np.zeros(glove_embeddings_size)

    
#Return the vector of a sentence
def sent2vec(sentence):
    global nlp
    if nlp is None:
        nlp = spacy.load('en')

    vec = word2vec('')
    if len(sentence) != 0:
        for w in nlp(sentence):
            if not (w.is_stop or w.is_punct):
                vec += word2vec(w.text)
    return vec   

#Compute the cosine similarity between two vectors
def cos_sim(vec1, vec2):
    return abs(cosine_similarity(vec1.reshape(1, -1), vec2.reshape(1, -1))[0][0])

############################### EXTRACTION/SIMILARITY ###############################

#Extract word vectors from text
def vector_extraction(passages):
    return [sent2vec(p) for p in passages]

#Extract entities from text
def entity_extraction(passages):

    global nlp, tokenizer
    if nlp is None:
        nlp = spacy.load('en')
    
    p_entities = []
    for p in passages:
        entities = []
        
        for v in hn_vocabulary:
            if v in p:
                entities.append(v)
                
        for e in nlp(p).ents:
            if e.text not in entities and e.label_ in ['PERSON', 'ORG', 'DATE', 'TIME', 'PERCENT', 'QUANTITY', 'CARDINAL']:
                entities.append(e.text)
        p_entities.append(list(set(entities)))
    
    return p_entities

#Extract topic from text
def topic_extraction(passages):
    return predict_topic(passages)

#Extract len from text
def len_extraction(passages):
    return [len(re.findall(r'\w+', p)) for p in passages]


def vector_similarity(vector_x, vector_y):
    return cos_sim(vector_x, vector_y)

def entity_similarity(set_x, set_y):
    return len(set_x.intersection(set_y)) / (len(set_x.union(set_y)) + 0.1)

def topic_similarity(tvx, tvy): 
    sim = 0.0
    #if both are not related to any topic return 0
    if not (len(set(tvx)) == len(set(tvy)) == 1 and set(tvx).pop() == set(tvy).pop() == 0.0625):    
        for tx, ty in zip(tvx, tvy):
            sim += (sqrt(tx) - sqrt(ty))**2
        sim = 1 - 1/sqrt(2) * sqrt(sim)
    return sim

def len_similarity(len_x, len_y):
    return min(len_x, len_y) / (max(len_x, len_y) + 0.1)


#Compute the cartesian similarity between the paragraphs of a pair of articles
def cartesian_similarity(pair):
    passages, vs, js, ls, ts = (0, ) * 5
    for vector_x, entities_x, topic_x, len_x in zip(pair['vector_x'], pair['entities_x'], pair['topic_x'], pair['len_x']):
        for vector_y, entities_y, topic_y, len_y in zip(pair['vector_y'], pair['entities_y'], pair['topic_y'], pair['len_y']):
            
            vs += vector_similarity(vector_x, vector_y)
            js += entity_similarity(set(entities_x), set(entities_y))
            ls += len_similarity(len_x, len_y)
            ts += topic_similarity(topic_x, topic_y)

            passages += 1

    similarity = [vs/passages, js/passages, ls/passages, ts/passages] if passages != 0  else [0, 0, 0, 0]
    
    print(similarity)
    return similarity

############################### PAIRS ###############################

#get scientific - news article (true and false) pairs
def split_training_test_set(graph_file, articles_file, set_type, model_folder):

    G = read_graph(graph_file)
    articles = pd.read_csv(articles_file, sep='\t')['url'].tolist()

    if set_type == 'test':
        pairs = []
        for a in articles:
            if G.out_degree(a) > 1:
                for p in G.successors(a):
                    pairs.append([a, p])
        
        df = pd.DataFrame(pairs, columns=['article', 'paper'])
        df.to_csv(model_folder+'/test_pairs.tsv', sep='\t', index=None)

    elif set_type == 'train':
        true_pairs = []
        for a in articles:
            if G.out_degree(a) == 1:
                true_pairs.append([a, next(iter(G.successors(a))), True])

        false_pairs = []
        for a in articles:
            if G.out_degree(a) == 1:
                true_successor = next(iter(G.successors(a)))
                while True:
                    index = randint(0, len(true_pairs)-1)
                    if true_pairs[index][1] != true_successor:
                        false_pairs.append([a, true_pairs[index][1], False])
                        break

        df = pd.DataFrame(true_pairs+false_pairs, columns=['article', 'paper', 'related'])
        df.to_csv(model_folder+'/train_pairs.tsv', sep='\t', index=None)


#Compute the similarity for each pair
def compute_passage_pairs(article_details_file, paper_details_file, set_type, model_folder):
    
    pairs = pd.read_csv(model_folder+'/'+set_type+'_pairs.tsv', sep='\t')
    article_details = pd.read_csv(article_details_file, sep='\t')
    paper_details = pd.read_csv(paper_details_file, sep='\t')

    for granularity in ['full_text', 'paragraph', 'sentence']:

        article_details['full_text'] = article_details['full_text'].apply(lambda x: split_text_to_passages(x, granularity))
        paper_details['full_text'] = paper_details['full_text'].apply(lambda x: split_text_to_passages(x, granularity))

        print('Extracting word vectors x')
        article_details['vector'] = article_details['full_text'].apply(vector_extraction)
        print('Extracting entities x')
        article_details['entities'] = article_details['full_text'].apply(entity_extraction)
        print('Extracting topics x')
        article_details['topic'] = article_details['full_text'].apply(topic_extraction)
        print('Extracting len x')
        article_details['len'] = article_details['full_text'].apply(len_extraction)

        print('Extracting word vectors y')
        paper_details['vector'] = paper_details['full_text'].apply(vector_extraction)
        print('Extracting entities y')
        paper_details['entities'] = paper_details['full_text'].apply(entity_extraction)
        print('Extracting topics y')
        paper_details['topic'] = paper_details['full_text'].apply(topic_extraction)
        print('Extracting len y')
        paper_details['len'] = paper_details['full_text'].apply(len_extraction)
        
        pairs = pairs.merge(paper_details[['url', 'vector', 'entities', 'topic', 'len']], left_on='paper', right_on='url')
        pairs = pairs.merge(article_details[['url', 'vector', 'entities', 'topic', 'len']], left_on='article', right_on='url')
        pairs = pairs[['paper', 'vector_x', 'entities_x', 'topic_x', 'len_x', 'article', 'vector_y', 'entities_y', 'topic_y', 'len_y', 'related']]

        df = pd.DataFrame(pairs.apply(cartesian_similarity, axis=1).tolist(), columns=['vec_sim', 'jac_sim', 'len_sim', 'top_sim']).reset_index(drop=True)
        pairs = pairs[['article', 'paper', 'related']].reset_index(drop=True)
        pairs = pd.concat([pairs, df], axis=1)    
        pairs.to_csv(model_folder+'/'+set_type+'_'+granularity+'.tsv', sep='\t', index=None)

################################ ####### ################################


#Classifier to test on multi-sourced articles
def predict_similarity(graph_file, articles_file, papers_file, model_folder):

    split_training_test_set(graph_file, articles_file, 'test', model_folder)
    compute_passage_pairs(articles_file, papers_file, 'test', model_folder)

    df1 = pd.read_csv(model_folder+'/train_full.tsv', sep='\t').rename(columns={'vec_sim': 'vec_sim_f', 'jac_sim': 'jac_sim_f', 'len_sim': 'len_sim_f', 'top_sim': 'top_sim_f'}).drop_duplicates()
    df2 = pd.read_csv(model_folder+'/train_paragraph.tsv', sep='\t').rename(columns={'vec_sim': 'vec_sim_p', 'jac_sim': 'jac_sim_p', 'len_sim': 'len_sim_p', 'top_sim': 'top_sim_p'}).drop_duplicates()
    df3 = pd.read_csv(model_folder+'/train_sentence.tsv', sep='\t').rename(columns={'vec_sim': 'vec_sim_p', 'jac_sim': 'jac_sim_p', 'len_sim': 'len_sim_p', 'top_sim': 'top_sim_p'}).drop_duplicates()

    df = df1.merge(df2, left_on=['article', 'paper', 'related'], right_on=['article', 'paper', 'related']).merge(df3, left_on=['article', 'paper', 'related'], right_on=['article', 'paper', 'related'])
    df = df.drop('related', axis=1)

    X = df[['vec_sim_f', 'jac_sim_f', 'len_sim_f', 'top_sim_f', 'vec_sim_p', 'jac_sim_p', 'len_sim_p', 'top_sim_p', 'vec_sim_s', 'jac_sim_s', 'len_sim_s', 'top_sim_s']].values
    
    classifier = pickle.load(open(model_folder+'/rf_model.sav', 'rb'))
    df['related'] = classifier.predict_proba(X)[:,1]

    df = df[['article', 'paper', 'related']]
    df.to_csv(model_folder+'/predict_pairs.tsv', sep='\t', index=None)

#Classifier to compute feature importance of similarities
def compute_similarity_model(graph_file, articles_file, papers_file, model_folder):
    classifier_type = 'RF'
    cross_val=True
    fold = 10
    n_est = 800
    m_dep = 200

    split_training_test_set(graph_file, articles_file, 'train', model_folder)
    compute_passage_pairs(articles_file, papers_file, 'train', model_folder)

    df1 = pd.read_csv(model_folder+'/train_full.tsv', sep='\t').rename(columns={'vec_sim': 'vec_sim_f', 'jac_sim': 'jac_sim_f', 'len_sim': 'len_sim_f', 'top_sim': 'top_sim_f'}).drop_duplicates()
    df2 = pd.read_csv(model_folder+'/train_paragraph.tsv', sep='\t').rename(columns={'vec_sim': 'vec_sim_p', 'jac_sim': 'jac_sim_p', 'len_sim': 'len_sim_p', 'top_sim': 'top_sim_p'}).drop_duplicates()
    df3 = pd.read_csv(model_folder+'/train_sentence.tsv', sep='\t').rename(columns={'vec_sim': 'vec_sim_p', 'jac_sim': 'jac_sim_p', 'len_sim': 'len_sim_p', 'top_sim': 'top_sim_p'}).drop_duplicates()

    df = df1.merge(df2, left_on=['article', 'paper', 'related'], right_on=['article', 'paper', 'related']).merge(df3, left_on=['article', 'paper', 'related'], right_on=['article', 'paper', 'related'])

    #clean some false positives
    df = df[~((df['related']==True) & ((df['vec_sim_f']==0) | (df['jac_sim_f']==0) | (df['top_sim_f']==0) | (df['vec_sim_p']==0) | (df['jac_sim_p']==0) | (df['top_sim_p']==0) | (df['vec_sim_s']==0) | (df['jac_sim_s']==0) | (df['top_sim_s']==0)))]

    #cross validation
    X = df[['vec_sim_f', 'jac_sim_f', 'len_sim_f', 'top_sim_f', 'vec_sim_p', 'jac_sim_p', 'len_sim_p', 'top_sim_p', 'vec_sim_s', 'jac_sim_s', 'len_sim_s', 'top_sim_s']].values
    #X = df[['vec_sim_f', 'jac_sim_f', 'len_sim_f', 'top_sim_f']].values
    y = df[['related']].values.ravel()

    fold = 5    
    if cross_val:
        kf = KFold(n_splits=fold, shuffle=True)
        score = 0.0
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            X_train = (X_train - X_train.mean(axis=0)) / X_train.std(axis=0)
            X_test = (X_test - X_test.mean(axis=0)) / X_test.std(axis=0)
            if classifier_type == 'RF':
                n_est = 800
                m_dep = 200
                #classifier = SVC()
                classifier = RandomForestClassifier(n_estimators=n_est, max_depth=m_dep, n_jobs=-1, random_state=42)
                classifier.fit(X_train, y_train)
                score += classifier.score(X_test, y_test)
            elif classifier_type == 'NN':

                num_epochs = 550000
                learning_rate = 1e-5
                hidden_layers = 120
                input_layers = 4
                # Logistic regression model
                model = nn.Sequential(
                nn.Linear(input_layers, hidden_layers),
                nn.BatchNorm1d(hidden_layers),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_layers, hidden_layers),
                nn.BatchNorm1d(hidden_layers),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_layers, 2)
                )
                # Loss and optimizer
                # nn.CrossEntropyLoss() computes softmax internally
                criterion = nn.BCEWithLogitsLoss()
                optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)  

                inputs = torch.Tensor(X_train)
                y_train = np.array([[0,1] if e == True else [1,0] for e in y_train.tolist()])
                targets = torch.Tensor(y_train).view(len(y_train), -1)

                # Train the model
                for epoch in range(num_epochs):
                                            
                    # Forward pass
                    outputs = model(inputs)
                    #print(outputs.dtype, targets.dtype, outputs.shape, targets.shape)
                    loss = criterion(outputs, targets)
                    
                    # Backward and optimize
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    if (epoch+1) % 100 == 0:
                        inputs_t = torch.Tensor(X_test)
                        labels_t = torch.Tensor(np.array([1 if e == True else 0 for e in y_test.tolist()]))
                        outputs_t = model(inputs_t)
                        _, predicted_t = torch.max(outputs_t.data, 1)
                        total = int(labels_t.size(0))
                        correct = int((predicted_t == labels_t).sum())
                        score = correct / total
                        print ('Epoch [{}/{}], Score: {:.4f}'.format(epoch+1, num_epochs, score))

                # Test the model
                # In test phase, we don't need to compute gradients (for memory efficiency)
                with torch.no_grad():

                    inputs = torch.Tensor(X_test)
                    labels = torch.Tensor(np.array([1 if e == True else 0 for e in y_test.tolist()]))
                    outputs = model(inputs)
                    _, predicted = torch.max(outputs.data, 1)
                    total = int(labels.size(0))
                    correct = int((predicted == labels).sum())
                    score += correct / total
                    print(score)

        print('Score:', score/fold)
        #print('Feature Importances:', classifier.feature_importances_)
    else:
        classifier = RandomForestClassifier(n_estimators=n_est, max_depth=m_dep)
        classifier.fit(X, y)
        pickle.dump(classifier, open(model_folder+'/rf_model.sav', 'wb'))
