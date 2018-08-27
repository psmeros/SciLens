import time
from random import randint

import networkx as nx
import numpy as np
import pandas as pd
from newspaper import Article

from diffusion_graph import read_graph
from settings import *
from textblob import TextBlob
from url_helpers import analyze_url, scrap_twitter_replies


def aggregate_tweet_details(graph_file, tweet_file, article_in_file, article_out_file):
    G = read_graph(graph_file)
    tweet_details = pd.read_csv(tweet_file, sep='\t')
    article_details = pd.read_csv(article_in_file, sep='\t')
    
    def func(url, tweet_details):
        dmax, dmin = pd.to_datetime(tweet_details['publish_date']).agg([np.max, np.min])
        delta_in_hours = (dmax-dmin).days *24 + (dmax-dmin).seconds // 3600
        agg = [url]
        agg.append(delta_in_hours)
        agg.append(len(tweet_details['user_country'].dropna().unique().tolist()))
        agg = agg + tweet_details[['RTs', 'replies_num', 'likes']].sum(axis=0).tolist()
        agg = agg + tweet_details[['tweet_polarity', 'tweet_subjectivity', 'replies_mean_polarity', 'replies_mean_subjectivity']].mean(axis=0).tolist()
        agg = agg + tweet_details[['user_followers_count', 'user_tweet_count', 'user_friends_count']].median(axis=0).tolist()

        agg = {'url':agg[0], 'tweets_time_delta':agg[1], 'users_countries':agg[2], 'retweets':agg[3], 'replies':agg[4], 'likes':agg[5], 'tweets_mean_polarity':agg[6], 'tweets_mean_subjectivity':agg[7], 'replies_mean_polarity':agg[8], 'replies_mean_subjectivity':agg[9], 'users_median_followers':agg[10], 'users_median_tweets':agg[11], 'users_median_friends':agg[12]}
        return agg

    article_details = article_details.merge(pd.DataFrame(article_details['url'].apply(lambda x: func(x, tweet_details[tweet_details['url'].isin(G.predecessors(x))])).tolist()), on='url')
    article_details.to_csv(article_out_file, sep='\t', index=None)


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


#get scientific - news article (true and false) pairs
def get_article_pairs(graph_file, articles_file, pairs_out_file):

    G = read_graph(graph_file)
    articles = open(articles_file).read().splitlines()

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
    df.to_csv(pairs_out_file, sep='\t', index=None)


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

#write documents to file
def get_effective_documents(graph_file, out_file, doc_type):
    G = read_graph(graph_file)

    documents = []
    if doc_type == 'articles':
        for d in G.predecessors(project_url+'#repository'):
            for p in G.predecessors(d):
                for a in G.predecessors(p):
                    if 'http://twitter.com' not in a:
                        documents.append(a)

    elif doc_type == 'tweets':
        for d in G.successors(project_url+'#twitter'):
            documents.append(d)

    with open(out_file, 'w') as f:
        for a in documents:
            f.write(a + '\n')

#prune the initial diffusion graph by keeping only the paths that contain the selected papers
def prune_graph(graph_in_file, graph_out_file, papers_file):

    if not useCache or not os.path.exists(graph_out_file):
        G = read_graph(graph_in_file)

        
        df = pd.read_csv(papers_file, sep='\t')
        df = df[~df['full_text'].isnull()]
        papers = df['url'].tolist()
        
        newG = nx.DiGraph()
        for path in nx.all_simple_paths(G, source=project_url+'#twitter', target=project_url+'#source'):
            for paper in papers:
                if paper in path:
                    for i in range(1, len(path)):
                        newG.add_edge(path[i-1], path[i])

        print(len([s for s in newG.successors(project_url+'#twitter')]), 'tweets out of', len(newG.nodes), 'nodes')
    
        with open(graph_out_file, 'w') as f:
            for edge in newG.edges:
                    f.write(edge[0] + '\t' + edge[1] + '\n')

    G = nx.DiGraph()
    edges = open(graph_out_file).read().splitlines()
    for e in edges:
        [e0, e1] = e.split()
        G.add_edge(e0, e1)

    return G

#download news/scientific article details
def download_articles(articles_file, out_file, sleep_time):
    articles = open(articles_file).read().splitlines()

    article_details=[]
    for i, a in enumerate(articles):
        print('\r %s%%' % ("{0:.2f}").format(100 * (i / float(len(articles)))), end = '\r')
        try:
            article = Article(a)
            article.download()
            article.parse()
            article.nlp()
            article_details.append([a, article.title, article.authors, article.keywords, article.publish_date, article.text])
            time.sleep(sleep_time)
        except:
            continue

    article_details = pd.DataFrame(article_details, columns=['url','title','authors','keywords','publish_date','full_text'])
    article_details.to_csv(out_file, sep='\t', index=None)

#get selected papers
def get_most_widely_referenced_publications(graph_file, out_file, num_of_domains):
    G = read_graph(graph_file)

    pubs = []
    for r in G.predecessors(project_url+'#repository'):
        for n in G.predecessors(r):
            domains = set()
            for w in G.predecessors(n):
                domain, _ = analyze_url(w)
                domains.add(domain)
            pubs.append([n, len(domains)])
    pubs = pd.DataFrame(pubs)
    pubs = pubs.sort_values(1, ascending=False)

    pubs[pubs[1]>=num_of_domains][0].to_csv(out_file, index=False)
