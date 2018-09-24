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

#remove articles that have the same text; tweets that point to that articles are redirected to one representative
def remove_duplicate_text(article_in_file, graph_in_file, article_out_file, graph_out_file):
    article_details = pd.read_csv(article_in_file, sep='\t')
    G = read_graph(graph_in_file)

    group = article_details.groupby('full_text')['url'].unique()
    duplicates = group[group.apply(lambda x: len(x)>1)].reset_index()['url'].tolist()

    remove_edges = []
    add_edges = []
    for d in duplicates:
        for l in d[1:]:
            for p in G.predecessors(l):
                remove_edges.append((p, l))
                add_edges.append((p, d[0]))
                article_details = article_details[article_details['url'] != l]

    for re in remove_edges:
        try:
            G.remove_edge(re[0], re[1])
        except:
            pass
    for ae in add_edges:
        G.add_edge(ae[0], ae[1])

    with open(graph_out_file, 'w') as f:
        for edge in G.edges:
            f.write(edge[0] + '\t' + edge[1] + '\n')

    article_details.to_csv(article_out_file, sep='\t', index=None)


#get scientific - news article (true and false) pairs
def get_article_pairs(graph_file, articles_file, pairs_out_file, set_type):

    G = read_graph(graph_file)
    articles = pd.read_csv(articles_file, sep='\t')['url'].tolist()

    if set_type == 'test':
        pairs = []
        for a in articles:
            if G.out_degree(a) > 1:
                for p in G.successors(a):
                    pairs.append([a, p])
        
        df = pd.DataFrame(pairs, columns=['article', 'paper'])
        df.to_csv(pairs_out_file, sep='\t', index=None)

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
        df.to_csv(pairs_out_file, sep='\t', index=None)


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
