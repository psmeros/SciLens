from settings import *
from quoteAnalysis import resolvePerson, resolveOrganization

#Plot functions

def plotQuotesAndTopicsDF(quotes, topics):
    
    topics = topics.toDF().toPandas()
    quotes = quotes.toDF().toPandas()
    topics = topics[(topics['articleSim']>topicSimThreshold) & (topics['quoteSim']>topicSimThreshold)]
    topics = quotes.merge(topics)

    df = pd.DataFrame()
    df['persons'] = topics.query("quoteeType == 'PERSON'").groupby(['articleTopic']).size()
    df['organizations'] = topics.query("quoteeType == 'ORG'").groupby(['articleTopic']).size()
    df['authority'] = topics.query("quoteeType == 'authority'").groupby(['articleTopic']).size()
    df['empirical observation'] = topics.query("quoteeType == 'empirical observation'").groupby(['articleTopic']).size()
    df = df.fillna(0)
    df = df.div(df.sum(axis=1), axis=0)
    df.to_csv('cache/'+sys._getframe().f_code.co_name+'_type.tsv', sep='\t')

    df = topics.query("quoteeType != 'unknown'")    
    df.to_csv('cache/'+sys._getframe().f_code.co_name+'_all.tsv', sep='\t')


def plotHeatMapDF(topics):
    
    topics = topics.toDF().toPandas()
    topics = topics[(topics['articleSim']>topicSimThreshold) & (topics['quoteSim']>topicSimThreshold)]

    df = topics[['articleTopic', 'quoteTopic']].groupby(['articleTopic', 'quoteTopic']).size().reset_index(name='counts').pivot(index='articleTopic', columns='quoteTopic', values='counts').fillna(0)
    df = df.div(df.sum(axis=1), axis=0)

    df.to_csv('cache/'+sys._getframe().f_code.co_name+'.tsv', sep='\t')

def plotTopQuoteesDF(quotes, topics):

    topics = topics.toDF().toPandas()
    quotes = quotes.toDF().toPandas()
    topics = topics[(topics['articleSim']>topicSimThreshold) & (topics['quoteSim']>topicSimThreshold)]
    topics = quotes.merge(topics)

    df_p = topics[topics['quoteeType'] == 'PERSON']['quotee'].value_counts().reset_index()
    df_p.columns = ['quotee', 'count']
    plist = df_p['quotee'].tolist()
    df_p['quotee'] = df_p['quotee'].apply(lambda x: resolvePerson(x, plist))
    df_p = df_p.groupby(['quotee']).sum().reset_index()
    df_p.columns = ['quotee', 'count']

    df_o = topics[topics['quoteeType'].isin(['ORG', 'PERSON'])]['quoteeAffiliation'].value_counts().reset_index()
    df_o.columns = ['organization', 'count']
    olist = df_o['organization'].tolist()
    df_o['organization'] = df_o['organization'].apply(lambda x: resolveOrganization(x, olist))
    df_o = df_o.groupby(['organization']).sum().reset_index()
    df_o.columns = ['organization', 'count']

    df_p.to_csv('cache/'+sys._getframe().f_code.co_name+'_p.tsv', sep='\t')
    df_o.to_csv('cache/'+sys._getframe().f_code.co_name+'_o.tsv', sep='\t')
