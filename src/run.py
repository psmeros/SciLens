from time import time
from diffusion_graph import create_graph
from graph_ops import download_articles, get_effective_documents, prune_graph, download_tweets
from quote_extraction import extract_quotes
from settings import twitterCorpusFile

t0 = time()

download_tweets(twitterCorpusFile, 'cache/top_paper_3_tweets.txt', 'cache/tweet_details.tsv', 1)


print("Total time: %0.3fs." % (time() - t0))
