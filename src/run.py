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

#second_level_graph()

#create graph
if not useCache or not os.path.exists(first_level_graph_file):
	first_level_graph()

df = pd.read_csv(first_level_graph_file, sep='\t')

G = nx.from_pandas_dataframe(df, 'tweet_url', 'out_url')

df = df.drop_duplicates('tweet_url').set_index('tweet_url')

for attr in ['timestamp', 'popularity', 'RTs', 'user_country']:
	nx.set_node_attributes(G, df[attr].to_dict(), attr)


#print(df['timestamp'])
print(G.nodes(data=True))



print("Total time: %0.3fs." % (time() - t0))
