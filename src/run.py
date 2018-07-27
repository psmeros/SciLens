from time import time
from diffusion_graph import create_graph
from graph_ops import download_articles, get_effective_documents, prune_graph, download_tweets, get_article_pairs
from settings import twitterCorpusFile, diffusion_graph_dir
from matching import prepare_articles_matching, create_annotation_subsample
from topic_detection import train_model, predict_topic

t0 = time()

#download_tweets(twitterCorpusFile, 'cache/top_paper_3_tweets.txt', 'cache/tweet_details.tsv', 1)
#prepare_articles_matching('cache/paper_details_v1.tsv', 'cache/paper_details_v2.tsv')
#prepare_articles_matching('cache/article_details_v1.tsv', 'cache/article_details_v2.tsv')
#get_article_pairs('pruned_graph_v2.tsv', 'article_pairs.tsv')
#train_model('cache/article_details_v2.tsv', 'cache/paper_details_v2.tsv', 'cache/topic_model')
#predict_topic('cache/article_details_v2.tsv', 'cache/article_details_v3.tsv', 'cache/paper_details_v2.tsv', 'cache/paper_details_v3.tsv', 'cache/topic_model')
create_annotation_subsample(diffusion_graph_dir+'article_pairs.tsv', 'cache/article_details_v3.tsv', 'cache/paper_details_v3.tsv', 2, 'cache/par_pairs_v1.tsv')

print("Total time: %0.3fs." % (time() - t0))
