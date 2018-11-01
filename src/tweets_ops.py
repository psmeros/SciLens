from random import randint

import networkx as nx
import numpy as np
import pandas as pd
from newspaper import Article
from textblob import TextBlob

from diffusion_graph import read_graph
from settings import *
from url_helpers import analyze_url, scrap_twitter_replies


def aggregate_tweet_details(graph_file, tweet_file, article_in_file, article_out_file):
    G = read_graph(graph_file)
    tweet_details = pd.read_csv(tweet_file, sep='\t').fillna(0)
    article_details = pd.read_csv(article_in_file, sep='\t')
    
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
    article_details.to_csv(article_out_file, sep='\t', index=None)

def prepare_annotation(tweets_file, out_file):
    df = pd.read_csv(tweets_file, sep='\t')
    df = df[['full_text', 'replies']]
    df['replies'] = df['replies'].apply(lambda x: np.NaN if not eval(x) else eval(x))
    df = df.dropna()

    df = df.join(df['replies'].apply(pd.Series).unstack().dropna().reset_index().drop('level_0', axis=1).set_index('level_1')).drop('replies', axis=1).rename(columns = {0:'reply'})
    df = df.sample(500)

    df.to_csv(out_file, sep='\t', index=None)

#extend tweet file
def extent_tweets(in_file, out_file):
    tweet_details = pd.read_csv(in_file, sep='\t')
    tweet_details['likes'] = tweet_details['popularity'] - tweet_details['RTs']
    tweet_details['tweet_polarity'] = tweet_details['full_text'].apply(lambda x: TextBlob(x).sentiment.polarity)
    tweet_details['tweet_subjectivity'] = tweet_details['full_text'].apply(lambda x: TextBlob(x).sentiment.subjectivity)
    tweet_details['replies_mean_polarity'] = tweet_details['replies'].apply(lambda x: np.mean([TextBlob(r).sentiment.polarity for r in eval(x)]))
    tweet_details['replies_mean_subjectivity'] = tweet_details['replies'].apply(lambda x: np.mean([TextBlob(r).sentiment.subjectivity for r in eval(x)]))
    tweet_details['user'] = tweet_details['url'].apply(lambda x: x.split('/')[3])
    tweet_details = tweet_details.merge(pd.read_csv(twitter_users_file, sep='\t'), left_on='user', right_on='screen_name', how='left')
    tweet_details = tweet_details.drop(['popularity', 'user', 'screen_name'], axis=1)
    tweet_details = tweet_details.replace('\\N',np.NaN)
    tweet_details.to_csv(out_file, sep='\t', index=None)



#write tweets to file
def download_tweets(corpus_file, in_file, out_file, sleep_time):

    df = pd.read_csv(twitterCorpusFile, sep='\t', header=None)
    df[0] = df[0].apply(lambda x: x.replace('https://','http://'))
    tweets_details = pd.read_csv(in_file, sep='\t', header=None)
    tweets_details = tweets_details.merge(df, left_on=0, right_on=0)
    tweets_details.columns = ['url','full_text','publish_date','popularity','RTs','user_country']
    
    tweets_details['replies'] = tweets_details['url'].apply(lambda x: scrap_twitter_replies(x, sleep_time))
    tweets_details['replies_num'] = tweets_details['replies'].apply(lambda x: len(x))
    tweets_details.to_csv(out_file, sep='\t', index=None)
