import re
import sys
import os.path
from time import time
import numpy as np
import pandas as pd
from spacy.en import English
from pyspark.sql import SparkSession
from pyspark import SparkConf

from settings import *

#Spark setup
def initSpark():
    global spark
    conf = SparkConf()
    conf.setAppName('quoteAnalysis')
    conf.setMaster('local[*]')
    conf.set('spark.executor.memory', memory)
    conf.set('spark.driver.memory', memory)
    conf.set('spark.driver.maxResultSize', '40G')
    conf.set('spark.jars.packages', 'org.postgresql:postgresql:42.1.4')
    spark = SparkSession.builder.config(conf=conf).getOrCreate()
    return spark

#Read/Write the results of *func* from/to cache
def cachefunc(func, args):
    
    cache = 'cache/'+func.__name__+'.parquet'
    if useCache and os.path.exists(cache):
        print ('Reading from cache:', cache)
        documents = spark.read.load(cache)
    else:
        t0 = time()
        documents = func(args)
        documents.write.parquet(cache, mode='overwrite')
        print(func.__name__, "ran in %0.3fs." % (time() - t0))
        
    return documents

#Create vocabulary for lowering dimensions
def createVocabulary():
    df = pd.read_csv(conceptsFile)[['Concept name', 'Concept literal']]
    df.columns = ['concept', 'literal']
    return df

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
def queryDB():

    #Database Settings
    user = dbSettings['user']
    password = dbSettings['password']
    db = dbSettings['db']
    host = dbSettings['host']
    port = dbSettings['port']

    query = """ (select (title || '.\n ' || body) as article
                from document
                where doc_type = 'web') doc """

    df = spark.read.jdbc("jdbc:postgresql://" + host + ':' + port + '/' + db, query,properties={"user": user, "password": password, "driver": "org.postgresql.Driver"})    
    df = df.limit(limitDocuments) if(limitDocuments!=-1) else df
    df = df.repartition(cores)

    return df


#Pretty print of numbers (by https://stackoverflow.com/a/45846841)
def human_format(num):
    num = float('{:.3g}'.format(num))
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    return '{}{}'.format('{:f}'.format(num).rstrip('0').rstrip('.'), ['', 'K', 'M', 'B', 'T'][magnitude])