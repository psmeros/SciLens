import numpy as np
import pandas as pd

from settings import *



# Load Data
print('Loading data')

con, meta = connectToDB('smeros', '', 'pizzaGate')

sql = '	select body, retweet_count, publishing_date \
		from document \
		where doc_type = \'twitter\''



def extractURLs(tweets):
	import re
	import requests
	import math

	urlRegex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

	urls = pd.Series(tweets['body'].apply(lambda x: re.findall(urlRegex, x)), name="urls")

	#a = urls.apply(lambda listUrl: for url in listUrl if url=='')
	resolvableURLs = urls.apply(pd.Series).applymap(lambda x: None if pd.isnull(x) else True if requests.get(x).status_code else False)
	resolvableURLs = pd.Series(np.logical_or.reduce(resolvableURLs, axis=1), name="hasResolvableURLs")
	print (resolvableURLs.shape)

	tweets = pd.concat([tweets, urls, resolvableURLs], axis=1)

	return tweets

#def hasResolvableURLs(tweets):



tweets = pd.read_sql(sql, con, index_col=None, coerce_float=True, params=None, parse_dates=None, columns=None, chunksize=None)

tweets = extractURLs(tweets)


#tweets['hasURL'] = np.where(hasResolvableURL(tweets['body']), True, False)

print (tweets[0:5])









#tweets = pd.DataFrame(read_file(POS_TWEETS_FILE), columns=['tweet'])
#tweets['sentiment'] = 1

# Data Shape
#print('\ttweets shape: ',tweets.shape)


