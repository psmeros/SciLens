from quoteAnalysis import quotePipeline
from preparePlots import *

documents, quotes, topics = quotePipeline()

#plotQuotesAndTopicsDF(quotes, topics)
#plotHeatMapDF(topics)
#plotTopQuoteesDF(quotes)