from urlAnalysis import first_level_graph, second_level_graph
from quoteAnalysis import extractQuotes
from topicAnalysis import discoverTopics
from preparePlots import *


t0 = time()


#documents, quotes = extractQuotes()
#topics =  discoverTopics(documents)

#plotQuotesAndTopicsDF(quotes, topics)
#plotHeatMapDF(topics)
#plotTopQuoteesDF(quotes, topics)

#first_level_graph()

second_level_graph()

print("Total time: %0.3fs." % (time() - t0))
