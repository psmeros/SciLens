from time import time
from diffusion_graph import create_graph
from graph_ops import download_articles, get_effective_documents, prune_graph

t0 = time()

get_effective_documents('pruned_graph_v2.tsv', 'cache/top_paper_3_tweets.txt', 'tweets')

print("Total time: %0.3fs." % (time() - t0))
