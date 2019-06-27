from pathlib import Path
import pickle
import re
from random import randint

import networkx as nx
import numpy as np
import pandas as pd
from newspaper import Article
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
from textblob import TextBlob

from create_corpus import read_graph

############################### CONSTANTS ###############################

scilens_dir = str(Path.home()) + '/Dropbox/scilens/'

stance_dir = scilens_dir + 'small_files/stance/'
semeval_tweets = stance_dir+'semeval_2016_stance.tsv'
annotated_tweets = stance_dir+'annotated_replies.csv'
stance_classifier = scilens_dir+'cache/stance_classifier.pkl'
twitter_users_file = scilens_dir + 'corpus/scilens_3M_users.tsv'

############################### ######### ###############################

################################ HELPERS ################################

def stance_features_extraction(df):
    df['word_count'] = df['Tweet'].apply(lambda x: len((re.sub(' +',' ',re.sub(r'[^a-zA-Z0-9 ]', '', x))).strip().split(' ')))

    df['negation'] = df['Tweet'].apply(lambda x: any(n in x for n in [' no ', ' not ', 'n\'t ']))

    positive_words = open(scilens_dir + 'small_files/opinion/positive-words.txt', encoding='utf-8', errors='ignore').read().splitlines()
    negative_words = open(scilens_dir + 'small_files/opinion/negative-words.txt', encoding='utf-8', errors='ignore').read().splitlines()
    df['positive'] = df['Tweet'].apply(lambda x: sum(n in x for n in positive_words))
    df['negative'] = df['Tweet'].apply(lambda x: sum(n in x for n in negative_words))

    df['length'] = df['Tweet'].apply(len)
    df['has_url'] = df['Tweet'].apply(lambda x: bool(re.search('http(s)?://', x)))
    df['quest_mark'] = df['Tweet'].apply(lambda x: x.count('?'))
    df['excl_mark'] = df['Tweet'].apply(lambda x: x.count('!'))

    df['tweet_polarity'] =  df['Tweet'].apply(lambda x: TextBlob(x).sentiment.polarity)
    df['tweet_subjectivity'] =  df['Tweet'].apply(lambda x: TextBlob(x).sentiment.subjectivity)
    sid = SentimentIntensityAnalyzer()
    df = df.join(df['Tweet'].apply(lambda x: sid.polarity_scores(x)).apply(pd.Series))

    return df

#train stance classifier based on semeval and crowd-annotated dataset
def train_stance_classifier():
    training_set_1 = pd.read_csv(semeval_tweets, sep='\t')
    training_set_1 = training_set_1[training_set_1['Target'].isin(['Climate Change is a Real Concern', 'Legalization of Abortion'])]
    training_set_1['Stance'] = training_set_1['Stance'].apply(lambda x: 'FAVOR-NEUTRAL' if x in ['FAVOR'] else x)
    training_set_1 = training_set_1[['Tweet', 'Stance']]
    #subsampling
    training_set_1 = pd.concat([training_set_1[training_set_1['Stance']=='FAVOR-NEUTRAL'].sample(training_set_1['Stance'].value_counts().min()), training_set_1[training_set_1['Stance']=='AGAINST'].sample(training_set_1['Stance'].value_counts().min())])

    training_set_2 = pd.read_csv(annotated_tweets).rename(columns={'what_do_you_believe_is_the_repliers_stance_position_towards_the_tweet': 'Stance', 'what_do_you_believe_is_the_repliers_stance_position_towards_the_tweet:confidence': 'confidence', 'reply':'Tweet'})
    training_set_2 = training_set_2[training_set_2.confidence>0.5][['Tweet', 'Stance']]
    training_set_2 = training_set_2[training_set_2['Stance']!='nr']

    training_set_2['Stance'] = training_set_2['Stance'].apply(lambda x: 'AGAINST' if x in ['con', 'quest'] else x).apply(lambda x: 'FAVOR-NEUTRAL' if x in ['com', 'sup']  else x)
    
    df = pd.concat([training_set_1, training_set_2])
    df = stance_features_extraction(df)

    X = np.array(df.drop(['Tweet', 'Stance'], axis=1).values, dtype=np.float32)
    y = df['Stance'].values
    
    n_est = 100
    m_dep = 20
    cross_validation = False

    if cross_validation:
        fold = 10
        kf = KFold(n_splits=fold, shuffle=True)
        score = 0.0
        for train_index, test_index in kf.split(X):

            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            classifier = RandomForestClassifier(n_estimators=n_est, max_depth=m_dep, n_jobs=-1)
            classifier.fit(X_train, y_train)

            score = classifier.score(X_test, y_test)
            conf_mat = confusion_matrix(y_test, classifier.predict(X_test))
            conf_mat = conf_mat.astype('float') / conf_mat.sum(axis=1)[:, np.newaxis]
            print(conf_mat)
            print('Score:', score)
    else:
        classifier = RandomForestClassifier(n_estimators=n_est, max_depth=m_dep, n_jobs=-1)
        classifier.fit(X, y)
        pickle.dump(classifier, open(stance_classifier, 'wb'))    

################################ ####### ################################

#attach social media details to article like tweet and replies sentiment, user features and replies' stance
def attach_social_media_details(graph_file, tweet_file, article_file):
    G = read_graph(graph_file)

    
    tweet_details = pd.read_csv(tweet_file, sep='\t')
    tweet_details['likes'] = tweet_details['popularity'] - tweet_details['RTs']
    tweet_details['tweet_polarity'] = tweet_details['full_text'].apply(lambda x: TextBlob(x).sentiment.polarity)
    tweet_details['tweet_subjectivity'] = tweet_details['full_text'].apply(lambda x: TextBlob(x).sentiment.subjectivity)
    tweet_details['replies_mean_polarity'] = tweet_details['replies'].apply(lambda x: np.mean([TextBlob(r).sentiment.polarity for r in eval(x)]))
    tweet_details['replies_mean_subjectivity'] = tweet_details['replies'].apply(lambda x: np.mean([TextBlob(r).sentiment.subjectivity for r in eval(x)]))
    tweet_details['user'] = tweet_details['url'].apply(lambda x: x.split('/')[3])
    tweet_details = tweet_details.merge(pd.read_csv(twitter_users_file, sep='\t'), left_on='user', right_on='screen_name', how='left')
    tweet_details = tweet_details.drop(['popularity', 'user', 'screen_name'], axis=1)
    tweet_details = tweet_details.replace('\\N',np.NaN)

    classifier = pickle.load(open(stance_classifier, 'rb'))
    X = np.array(stance_features_extraction(tweet_details[['replies']].rename(columns={'replies':'Tweet'})).drop('Tweet', axis=1).values, dtype=np.float32)
    tweet_details['stance']=classifier.predict_proba(X)[:,1]
    tweet_details['stance']= tweet_details.apply(lambda x: 0 if x.replies_num==0 else x.stance, axis=1)
    tweet_details = tweet_details.fillna(0)

    article_details = pd.read_csv(article_file, sep='\t')
    
    def func(url, tweet_details):
        tweet_details = tweet_details.copy()
        if len(tweet_details['publish_date']) in [0, 1]:
            delta_in_hours = 0
        else:
            tweet_details['publish_date'] = pd.to_datetime(tweet_details['publish_date']).astype('int64')//1e9
            [dmin, dmax] = np.percentile(tweet_details['publish_date'].tolist(), [5, 95])
            delta_in_hours = (dmax-dmin) // 3600
            if len(tweet_details['publish_date'])!=2:
                tweet_details = tweet_details[(dmin<tweet_details['publish_date']) & (tweet_details['publish_date']<dmax)]
    
        agg = [url]
        agg.append(delta_in_hours)
        agg.append(len(tweet_details['user_country'].dropna().unique().tolist()))
        agg = agg + tweet_details[['RTs', 'replies_num', 'likes']].sum(axis=0).tolist()
        agg = agg + tweet_details[['tweet_polarity', 'tweet_subjectivity', 'replies_mean_polarity', 'replies_mean_subjectivity']].mean(axis=0).tolist()
        agg = agg + tweet_details[['user_followers_count', 'user_tweet_count', 'user_friends_count']].median(axis=0).tolist()
        agg.append(tweet_details['stance'].mean(axis=0))

        agg = {'url':agg[0], 'tweets_time_delta':agg[1], 'users_countries':agg[2], 'retweets':agg[3], 'replies_count':agg[4], 'likes':agg[5], 'tweets_mean_polarity':agg[6], 'tweets_mean_subjectivity':agg[7], 'replies_mean_polarity':agg[8], 'replies_mean_subjectivity':agg[9], 'users_median_followers':agg[10], 'users_median_tweets':agg[11], 'users_median_friends':agg[12], 'stance':agg[13]}
        return agg

    article_details = article_details.merge(pd.DataFrame(article_details['url'].apply(lambda x: func(x, tweet_details[tweet_details['url'].isin(G.predecessors(x))])).tolist()), on='url')
    return article_details
