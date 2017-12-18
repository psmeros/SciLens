from settings import *
from quoteAnalysis import quotePipeline

#Plot functions
def plotNumOfQuotesDF():
    dfPath = 'cache/plotNumOfQuotesDF.pkl'
    if os.path.exists(dfPath):
        return pd.read_pickle(dfPath)
    
    documents, quotes, topics = quotePipeline()
    topics = topics.toDF().toPandas()
    quotes = quotes.toDF().toPandas()
    topics = topics[(topics['articleSim']>0.5) & (topics['quoteSim']>0.5)]
    topics = quotes.merge(topics)

    df = pd.DataFrame()
    df['persons'] = topics.query("quoteeType == 'PERSON'").groupby(['articleTopic']).size()
    df['organizations'] = topics.query("quoteeType == 'ORG'").groupby(['articleTopic']).size()
    df['authority'] = topics.query("quotee == 'authority'").groupby(['articleTopic']).size()
    df['empirical observation'] = topics.query("quotee == 'empirical observation'").groupby(['articleTopic']).size()
    df = df.fillna(0)
    df = df.div(df.sum(axis=1), axis=0)
    df.to_pickle(dfPath)
    return df
