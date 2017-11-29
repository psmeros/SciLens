import re
import sys
import os.path
from time import time
import numpy as np
import pandas as pd
from spacy.en import English
from pyspark.sql import SparkSession
from pyspark.sql import SQLContext
from pyspark import SparkConf, SparkContext 

from pyspark.sql.types import *

from settings import *

#Create Keyword Lists
nlp = English()
authorityKeywords = [nlp(x)[0].lemma_ for x in ['expert', 'scientist', 'researcher', 'professor', 'author', 'paper', 'report', 'study', 'analysis', 'research', 'survey', 'release']]
empiricalKeywords = [nlp(x)[0].lemma_ for x in ['study', 'people']]
actionsKeywords = [nlp(x)[0].lemma_ for x in ['prove', 'demonstrate', 'reveal', 'state', 'mention', 'report', 'say', 'show', 'announce', 'claim', 'suggest', 'argue', 'predict', 'believe', 'think']]


#Spark setup
spark = SparkSession.builder.appName('quoteAnalysis').master("local[*]").config('spark.jars.packages', 'org.postgresql:postgresql:42.1.4').getOrCreate()
spark.conf.set('spark.executor.memory', '2G')
spark.conf.set('spark.driver.memory', '5G')
spark.conf.set('spark.driver.maxResultSize', '10G')


#Read/Write the results of *func* from/to cache
def cachefunc(func, args):
    cache = 'cache/'+func.__name__+'.pkl'
    if useCache and os.path.exists(cache):
        print ('Reading from cache:', cache)
        documents = pd.read_pickle(cache)
    else:
        t0 = time()
        documents = func(args)
        print(func.__name__, "ran in %0.3fs." % (time() - t0))
        #documents.to_pickle(cache)
    return documents

#Create vocabulary for the Vectorizer
def createVocabulary():
    return pd.read_csv(conceptsFile)['Concept literal'].tolist()

#Transform tf matrix (map to lower dimensions)
def transformTF(tf):

    concepts = pd.read_csv(conceptsFile)['Concept name'].tolist()

    boundaries = []
    labels = []
    start = 0
    concept = concepts[0]

    for end,c in enumerate(concepts):
        if c != concept:
            boundaries.append((start, end))
            labels.append(concept)
            start = end
            concept = c

    labels.append(concept)
    boundaries.append((start, end+1))

    tf = tf.T
    new_tf = np.empty((len(labels), tf.shape[1]))
    
    for i, b in enumerate(boundaries): 
        (start, end) = b
        new_tf[i] = tf[start:end].sum(axis=0)

    return new_tf.T, labels


#Pose a query to the DB
def queryDB(doc_type=''):

    #Database Settings
    user = dbSettings['user']
    password = dbSettings['password']
    db = dbSettings['db']
    host = dbSettings['host']
    port = dbSettings['port']

    if (doc_type=='web'):
        query = """
        (select title, body
        from document
        where doc_type = 'web') doc"""
    else:
        query = """
        (select body, doc_type
        from document
        where doc_type = 'web')
        UNION
        (select body, doc_type
        from document
        where doc_type = 'twitter') """


    df = spark.read.jdbc("jdbc:postgresql://" + host + ':' + port + '/' + db, query,properties={"user": user, "password": password, "driver": "org.postgresql.Driver"})    
    df = df.limit(limitDocuments) if(limitDocuments!=-1) else df

    return df




#Pretty print of numbers (by https://stackoverflow.com/a/45846841)
def human_format(num):
    num = float('{:.3g}'.format(num))
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    return '{}{}'.format('{:f}'.format(num).rstrip('0').rstrip('.'), ['', 'K', 'M', 'B', 'T'][magnitude])