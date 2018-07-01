from time import time
from diffusion_graph import create_graph
from quoteAnalysis import extractQuotes
from topicAnalysis import discoverTopics
from plots import *
from utils import create_crawl_keywords


t0 = time()


#documents, quotes = extractQuotes()
#topics =  discoverTopics(documents)

#plotQuotesAndTopicsDF(quotes, topics)
#plotHeatMapDF(topics)
#plotTopQuoteesDF(quotes, topics)

create_graph()
#create_crawl_keywords()


print("Total time: %0.3fs." % (time() - t0))
