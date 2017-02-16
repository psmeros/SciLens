import numpy as np
import pandas as pd

from settings import *



# Load Data
print('Loading data')

con, meta = connectToDB('smeros', '', 'pizzaGate')

sql = '	select body, retweet_count, publishing_date \
		from document \
		where doc_type = \'twitter\''



def hasResolvableURL(tweetBody):
    tweetBody.str.extract("(?P<letter>[a-z])(?P<digit>[0-9])", expand=True)
    print (tweetBody)



tweets = pd.read_sql(sql, con, index_col=None, coerce_float=True, params=None, parse_dates=None, columns=None, chunksize=None)

hasResolvableURL(tweets['body'])
#tweets['hasURL'] = np.where(hasResolvableURL(tweets['body']), True, False)

print (tweets[0:5])









#tweets = pd.DataFrame(read_file(POS_TWEETS_FILE), columns=['tweet'])
#tweets['sentiment'] = 1

# Data Shape
#print('\ttweets shape: ',tweets.shape)


