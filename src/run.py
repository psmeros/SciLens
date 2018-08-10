from time import time
from diffusion_graph import create_graph
from graph_ops import download_articles, get_effective_documents, prune_graph, download_tweets, get_article_pairs, extent_tweets, aggregate_tweet_details
from settings import twitterCorpusFile, cache_dir
from matching import prepare_articles_matching, compute_pairs_similarity, uniformly_random_subsample, compute_similarity_model
from topic_detection import train_model, predict_topic

t0 = time()

#download_tweets(twitterCorpusFile, cache_dir + 'top_paper_3_tweets.txt', cache_dir + 'tweet_details_v1.tsv', 1)
#prepare_articles_matching(cache_dir + 'paper_details_v1.tsv', cache_dir + 'paper_details_v2.tsv')
#prepare_articles_matching(cache_dir + 'article_details_v1.tsv', cache_dir + 'article_details_v2.tsv')
#train_model(cache_dir + 'article_details_v2.tsv', cache_dir + 'paper_details_v2.tsv', cache_dir + 'topic_model')
#predict_topic(cache_dir + 'article_details_v2.tsv', cache_dir + 'article_details_v3.tsv', cache_dir + 'paper_details_v2.tsv', cache_dir + 'paper_details_v3.tsv', cache_dir + 'topic_model')
#uniformly_random_subsample(cache_dir + 'par_pairs_v1.tsv', 200, cache_dir + 'par_pairs_v2.tsv')
#extent_tweets(cache_dir+'tweet_details_v1.tsv', cache_dir+'tweet_details_v2.tsv')
#aggregate_tweet_details(cache_dir+'diffusion_graph/pruned_graph_v2.tsv', cache_dir+'tweet_details_v2.tsv', cache_dir+'article_details_v3.tsv', cache_dir+'article_details_v4.tsv')
#get_article_pairs(cache_dir+'diffusion_graph/pruned_graph_v2.tsv', cache_dir+'articles.txt', cache_dir+'pairs.tsv')
#compute_pairs_similarity(cache_dir + 'similarity_model_pairs_v1.tsv', cache_dir + 'article_details_v3.tsv', cache_dir + 'paper_details_v3.tsv', cache_dir + 'similarity_model_pairs_v2.tsv')
compute_similarity_model(cache_dir+'similarity_model_pairs_v2.tsv')
print("Total time: %0.3fs." % (time() - t0))
