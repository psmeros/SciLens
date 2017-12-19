from settings import *
from quoteAnalysis import quotePipeline

#Plot functions

def plotNumOfQuotesDF():
    dfPath = 'cache/'+sys._getframe().f_code.co_name+'.pkl'
    if os.path.exists(dfPath):
        return pd.read_pickle(dfPath)
    
    documents, quotes, topics = quotePipeline()
    topics = topics.toDF().toPandas()
    quotes = quotes.toDF().toPandas()
    topics = topics[(topics['articleSim']>topicSimThreshold) & (topics['quoteSim']>topicSimThreshold)]
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


def plotHeatMapDF():
    dfPath = 'cache/'+sys._getframe().f_code.co_name+'.pkl'
    if os.path.exists(dfPath):
        return pd.read_pickle(dfPath)
    
    documents, quotes, topics = quotePipeline()
    topics = topics.toDF().toPandas()
    topics = topics[(topics['articleSim']>topicSimThreshold) & (topics['quoteSim']>topicSimThreshold)]

    df = topics[['articleTopic', 'quoteTopic']].groupby(['articleTopic', 'quoteTopic']).size().reset_index(name='counts').pivot(index='articleTopic', columns='quoteTopic', values='counts').fillna(0)
    df = df.div(df.sum(axis=1), axis=0)

    df.to_pickle(dfPath)
    return df

def plotTopQuoteesDF():
    dfPath_p = 'cache/'+sys._getframe().f_code.co_name+'_p.pkl'
    dfPath_o = 'cache/'+sys._getframe().f_code.co_name+'_o.pkl'
    if os.path.exists(dfPath_p) and os.path.exists(dfPath_o):
        return pd.read_pickle(dfPath_p), pd.read_pickle(dfPath_o)

    documents, quotes, topics = quotePipeline()
    quotes = quotes.toDF().toPandas()

    df_p = quotes[quotes['quoteeType'] == 'PERSON']['quotee'].value_counts().reset_index()
    df_p.columns = ['quotee', 'count']

    df_o = quotes[quotes['quoteeType'] == 'ORG']['quoteeAffiliation'].value_counts().reset_index()
    df_o.columns = ['organization', 'count']

    df_p.to_pickle(dfPath_p)
    df_o.to_pickle(dfPath_o)
    return df_p, df_o