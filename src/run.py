from time import time
from diffusion_graph import create_graph
from graph_ops import download_articles, get_effective_documents, prune_graph, download_tweets, get_article_pairs
from quote_extraction import extract_quotes
from settings import twitterCorpusFile, diffusion_graph_dir
from matching import prepare_articles_matching

t0 = time()

#download_tweets(twitterCorpusFile, 'cache/top_paper_3_tweets.txt', 'cache/tweet_details.tsv', 1)
prepare_articles_matching('cache/paper_details_v1.tsv', 'cache/paper_details_v2.tsv')
prepare_articles_matching('cache/article_details_v1.tsv', 'cache/article_details_v2.tsv')
#get_article_pairs('pruned_graph_v2.tsv', 'article_pairs.tsv')

print("Total time: %0.3fs." % (time() - t0))
