from time import time

#from graph_ops import *
#from matching import *
#from quote_extraction import *
#from settings import *
#from topic_detection import *
#from tweets_ops import *
from diffusion_graph import *
t0 = time()

#download_tweets(twitterCorpusFile, cache_dir + 'top_paper_3_tweets.txt', cache_dir + 'tweet_details_v1.tsv', 1)
#train_topic_model(cache_dir + 'article_details_v2.tsv', cache_dir + 'paper_details_v2.tsv', cache_dir + 'topic_model')
#extent_tweets(cache_dir+'tweet_details_v1.tsv', cache_dir+'tweet_details_v2.tsv')
#aggregate_tweet_details(cache_dir+'diffusion_graph/pruned_graph_v3.tsv', cache_dir+'tweet_details_v2.tsv', cache_dir+'article_details_v5.tsv', cache_dir+'article_details_v6.tsv')
#get_article_pairs(cache_dir+'diffusion_graph/pruned_graph_v3.tsv', cache_dir+'article_details_v5.tsv', cache_dir+'pairs.tsv', True)
#compute_pairs_similarity(cache_dir + 'similarity_model_test_pairs_v1.tsv', cache_dir + 'article_details_v5.tsv', cache_dir + 'paper_details_v3.tsv', 'full_text', cache_dir + 'similarity_model_pairs_test_v2_full.tsv')
#compute_pairs_similarity(cache_dir + 'similarity_model_test_pairs_v1.tsv', cache_dir + 'article_details_v5.tsv', cache_dir + 'paper_details_v3.tsv', 'paragraph', cache_dir + 'similarity_model_pairs_test_v2_paragraph.tsv')
#compute_pairs_similarity(cache_dir + 'similarity_model_test_pairs_v1.tsv', cache_dir + 'article_details_v5.tsv', cache_dir + 'paper_details_v3.tsv', 'sentence', cache_dir + 'similarity_model_pairs_test_v2_sentence.tsv')
#compute_similarity_model(cache_dir+'similarity_model/train_pairs_v2', 'RF', cache_dir+'similarity_model/rf_model.sav', True)
#extract_quotes(cache_dir+'article_details_v3.tsv', cache_dir+'article_details_v4.tsv')
#remove_duplicate_text(cache_dir+'article_details_v4.tsv', cache_dir+'diffusion_graph/pruned_graph_v2.tsv', cache_dir+'article_details_v5.tsv', cache_dir+'diffusion_graph/pruned_graph_v3.tsv') 
#test_similarity_model(cache_dir+'similarity_model/test_pairs_v2', cache_dir+'similarity_model/rf_model.sav', cache_dir+'similarity_model/test_pairs_v3.tsv')

#prepare_annotation(cache_dir+'tweet_details_v2.tsv', cache_dir+'tweet_replies.tsv')



diffusion_graph_dir = scilens_dir + 'cache/test_diffusion_graph/'

#create_diffusion_graph(scilens_dir + 'corpus/sample.tsv', diffusion_graph_dir + 'sample_v1.tsv')
#clean_orphan_nodes(diffusion_graph_dir + 'sample_v1.tsv', diffusion_graph_dir + 'sample_v2.tsv')
#download_documents(diffusion_graph_dir + 'sample_v2.tsv', diffusion_graph_dir + 'sample_v3.tsv', diffusion_graph_dir + 'paper_details_v1.tsv', 'paper')
#clean_orphan_nodes(diffusion_graph_dir + 'sample_v3.tsv', diffusion_graph_dir + 'sample_v4.tsv')
#download_documents(diffusion_graph_dir + 'sample_v4.tsv', diffusion_graph_dir + 'sample_v5.tsv', diffusion_graph_dir + 'article_details_v1.tsv', 'article')
#clean_orphan_nodes(diffusion_graph_dir + 'sample_v5.tsv', diffusion_graph_dir + 'sample_v6.tsv')
#download_tweets(diffusion_graph_dir + 'sample_v6.tsv', scilens_dir + 'corpus/sample.tsv', diffusion_graph_dir + 'tweet_details_v1.tsv')
print("Total time: %0.3fs." % (time() - t0))
