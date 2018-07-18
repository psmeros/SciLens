from time import time
from diffusion_graph import create_graph
from graph_ops import download_articles, get_effective_documents, prune_graph
from quote_extraction import extract_quotes

t0 = time()

extract_quotes('cache/article_details_small.tsv', 'cache/article_details_2.tsv')


print("Total time: %0.3fs." % (time() - t0))
