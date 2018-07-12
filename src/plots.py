import sys
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
import networkx as nx
from networkx.algorithms import bipartite

from settings import *
from quoteAnalysis import resolvePerson, resolveOrganization
from url_helpers import analyze_url


#Grey vs Scientific Literature
def plot_papers_info(papers_file):
    df = pd.read_csv(papers_file,header=None)
    df['domain'] = df.apply(lambda x: analyze_url(x[0])[0], axis=1)
    df['grey?'] = df['domain'].apply(lambda x: 'grey' if '.gov' in x else 'non-grey')
    df['pdf?'] = df[0].apply(lambda x: 'pdf' if '.pdf' in x else 'html')
    print(df.groupby('grey?').size().div(df.shape[0]))
    print(df.groupby('pdf?').size().div(df.shape[0]))
    _, axarr = plt.subplots(2)
    tmp = pd.DataFrame(df[df['domain'].str.contains('.gov')]['domain'].apply(lambda x:(lambda x: x[-2]+'.'+x[-1])(x.split(".")))).groupby('domain').size().sort_values(ascending=False)
    tmp[tmp>4].plot('barh' , colormap='RdBu', ax=axarr[0], figsize=(10,10))
    tmp = pd.DataFrame(df[~df['domain'].str.contains('.gov')]['domain'].apply(lambda x:(lambda x: x[-2]+'.'+x[-1])(x.split(".")))).groupby('domain').size().sort_values(ascending=False)
    tmp[tmp>1].plot('barh', colormap='RdBu', ax=axarr[1], figsize=(10,10))
    plt.draw()

#Plot helpers
def plot_helper(inst, repos, countries):
    df = pd.read_csv(diffusion_graph_dir+'epoch_0.tsv', sep='\t').dropna()    
    tweets = df.copy().drop('target_url', axis=1).drop_duplicates('source_url')
    #beutify country names
    tweets = tweets.merge(pd.read_csv(countriesFile).rename(columns={'Name':'Country'}), left_on='user_country', right_on='Code').drop(['user_country', 'Code'], axis=1).set_index('source_url')
    tweets.loc[tweets['Country'] == 'United States', 'Country'] = 'USA'
    print('Initial Tweets:', len(tweets))

    #Popularity
    inst.groupby('Institution').mean()['popularity'].sort_values(ascending=False)[:20]
    repos.groupby('Field').size().sort_values(ascending=False)
    inst.groupby('Institution').mean().plot.scatter(x='Score', y='popularity')
    corr = inst.groupby('Institution').mean()[['popularity', 'World Rank', 'National Rank', 'Alumni Employment', 'Publications', 'Influence', 'Citations', 'Broad Impact', 'Patents', 'Score']].corr()
    #sns.heatmap(corr, xticklabels=corr.columns.values, yticklabels=corr.columns.values)
    corr.iloc[0]

    #bipartite graph
    countries['Name'] = countries['Name'].map(lambda n: n+'_user')
    countries['Location'] = countries['Location'].map(lambda n: n+'_inst')
    B = nx.Graph()
    B.add_edges_from([(row['Name'], row['Location']) for _, row in countries.iterrows()])
    plt.figure(figsize=(10,10))
    X, Y = bipartite.sets(B)
    pos = dict()
    pos.update( (n, (1, i)) for i, n in enumerate(X) ) # put nodes from X at x=1
    pos.update( (n, (2, i*4)) for i, n in enumerate(Y) ) # put nodes from Y at x=2
    nx.draw(B, pos=pos, with_labels = True)

#Plot URL decay per year
def plot_URL_decay(graph_nodes):
    df = pd.read_csv(diffusion_graph_dir+'epoch_0.tsv', sep='\t')
    df['date'] = df['timestamp'].apply(lambda s : datetime.strptime(s, '%Y-%m-%d %H:%M:%S').year)
    df['target_url'] = df['target_url'].apply(lambda u: u if u in [graph_nodes['tweetWithoutURL'], graph_nodes['HTTPError'], graph_nodes['TimeoutError']] else 'working URL')
    df['Tweets with'] = df['target_url'].map(lambda n: 'HTTP error in outgoing URL' if n == graph_nodes['HTTPError'] else 'timeout error in outgoing URL' if n == graph_nodes['TimeoutError'] else 'no URL' if n == graph_nodes['tweetWithoutURL'] else 'working URL')
    df[['source_url', 'date','Tweets with']].pivot_table(index='date', columns='Tweets with',aggfunc='count').T.reset_index(level=0, drop=True).T.fillna(1).plot(logy=True, figsize=(10,10), sort_columns=True)


#Tableau plots
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
