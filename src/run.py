from urlAnalysis import first_level_graph
from quoteAnalysis import extractQuotes
from topicAnalysis import discoverTopics
from preparePlots import *
from scraping import scrap_nutritionfacts


t0 = time()


documents, quotes = extractQuotes()
topics =  discoverTopics(documents)

#plotQuotesAndTopicsDF(quotes, topics)
#plotHeatMapDF(topics)
#plotTopQuoteesDF(quotes, topics)

#first_level_graph()

scrap_nutritionfacts()

print("Total time: %0.3fs." % (time() - t0))
