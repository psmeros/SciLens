import time

import networkx as nx
import pandas as pd
from textblob import TextBlob

from diffusion_graph import read_graph
from newspaper import Article
from settings import *
from url_helpers import analyze_url, scrap_twitter_replies


def aggregate_tweets(graph_file, article_file, tweet_file, attribute):
    G = read_graph(graph_file)


def extent_tweets(tweet_file):
    tweet_details = pd.read_csv(tweet_file, sep='\t')
    tweet_details['likes'] = tweet_details['popularity'] - tweet_details['RTs']
    tweet_details['tweet_polarity'] = tweet_details['full_text'].apply(lambda x: TextBlob(x).sentiment.polarity)
    tweet_details['tweet_subjectivity'] = tweet_details['full_text'].apply(lambda x: TextBlob(x).sentiment.subjectivity)
    tweet_details['replies_mean_polarity'] = tweet_details['replies'].apply(lambda x: np.mean([TextBlob(r).sentiment.polarity for r in eval(x)]))
    tweet_details['replies_mean_subjectivity'] = tweet_details['replies'].apply(lambda x: np.mean([TextBlob(r).sentiment.subjectivity for r in eval(x)]))

#write scientific - news article pairs
def get_article_pairs(graph_in_file, graph_out_file):
    
    G = read_graph(graph_in_file)

    pairs = []
    for d in G.predecessors(project_url+'#repository'):
        for p in G.predecessors(d):
            for a in G.predecessors(p):
                if 'http://twitter.com' not in a:
                    pairs.append([p, a])

    with open(graph_out_file, 'w') as f:
        f.write('paper\tarticle\n')
        for p in pairs:
            f.write(p[0] + '\t' +p[1] + '\n')


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
