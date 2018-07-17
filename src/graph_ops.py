import time

import networkx as nx
import pandas as pd
from newspaper import Article

from diffusion_graph import read_graph
from settings import *
from url_helpers import analyze_url



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

    if not useCache or not os.path.exists(diffusion_graph_dir+graph_out_file):
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
    
        with open(diffusion_graph_dir+graph_out_file, 'w') as f:
            for edge in newG.edges:
                    f.write(edge[0] + '\t' + edge[1] + '\n')

    G = nx.DiGraph()
    edges = open(diffusion_graph_dir+graph_out_file).read().splitlines()
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

#(Deprecated)
def get_most_popular_publications(graph_file, out_file):
    G = read_graph(graph_file)
    
    df = pd.read_csv(diffusion_graph_dir+'epoch_0.tsv', sep='\t').dropna()

    for _, row in df.iterrows():
        G.add_node(row['source_url'], popularity=row['popularity'], timestamp=row['timestamp'], user_country=row['user_country'])

    pubs = []
    for r in G.predecessors(project_url+'#repository'):
        for n in G.predecessors(r):
            popularity = 0
            for path in nx.all_simple_paths(G, source=project_url+'#twitter', target=n):
                popularity += G.node[path[1]]['popularity']
            pubs.append([n , popularity])

    pubs = pd.DataFrame(pubs)
    pubs = pubs.sort_values(1, ascending=False)
    pubs[0].to_csv(out_file, index=False)
