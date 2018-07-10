import networkx as nx
import pandas as pd

from diffusion_graph import create_graph
from settings import *
from url_helpers import analyze_url



def get_most_widely_referenced_publications(different_domains, filename):
    G = create_graph()

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

    pubs[pubs[1]>=different_domains][0].to_csv(filename, index=False)

#(Deprecated)
def get_most_popular_publications(filename):
    G = create_graph()
    
    df = pd.read_csv(diffusion_graph_dir+'epoch_0.tsv', sep='\t').dropna()
    df['social'] = project_url+'#twitter'
    G =  nx.compose(G, nx.from_pandas_edgelist(df, source='social', target='source_url', create_using=nx.DiGraph()))

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
    pubs[0].to_csv(filename, index=False)
