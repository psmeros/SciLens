from time import time
from settings import *
from graph_ops import *
from matching import *
from topic_detection import *
from quote_extraction import *

t0 = time()

#download_tweets(twitterCorpusFile, cache_dir + 'top_paper_3_tweets.txt', cache_dir + 'tweet_details_v1.tsv', 1)
#train_topic_model(cache_dir + 'article_details_v2.tsv', cache_dir + 'paper_details_v2.tsv', cache_dir + 'topic_model')
#uniformly_random_subsample(cache_dir + 'par_pairs_v1.tsv', 200, cache_dir + 'par_pairs_v2.tsv')
#extent_tweets(cache_dir+'tweet_details_v1.tsv', cache_dir+'tweet_details_v2.tsv')
#aggregate_tweet_details(cache_dir+'diffusion_graph/pruned_graph_v2.tsv', cache_dir+'tweet_details_v2.tsv', cache_dir+'article_details_v3.tsv', cache_dir+'article_details_v4.tsv')
#get_article_pairs(cache_dir+'diffusion_graph/pruned_graph_v2.tsv', cache_dir+'articles.txt', cache_dir+'pairs.tsv')
compute_pairs_similarity(cache_dir + 'similarity_model_pairs_v1_sample.tsv', cache_dir + 'article_details_v5.tsv', cache_dir + 'paper_details_v3.tsv', 'full_text', cache_dir + 'similarity_model_pairs_v2_sample.tsv')
#compute_similarity_model(cache_dir+'similarity_model_pairs_v2.tsv', cache_dir+'rf_model.sav', True)
#extract_quotes(cache_dir+'article_details_v3.tsv', cache_dir+'article_details_v4.tsv')
print("Total time: %0.3fs." % (time() - t0))
