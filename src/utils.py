import sqlalchemy
import warnings
import re
import sys
import os.path
from time import time
import numpy as np
from spacy.en import English

from settings import *

#Create Keyword Lists
nlp = English()
authorityKeywords = [nlp(x)[0].lemma_ for x in ['expert', 'scientist', 'researcher', 'professor', 'author', 'paper', 'report', 'study', 'analysis', 'research', 'survey', 'release']]
empiricalKeywords = [nlp(x)[0].lemma_ for x in ['study', 'people']]
actionsKeywords = [nlp(x)[0].lemma_ for x in ['prove', 'demonstrate', 'reveal', 'state', 'mention', 'report', 'say', 'show', 'announce', 'claim', 'suggest', 'argue', 'predict', 'believe', 'think']]

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
        documents.to_pickle(cache)
    return documents

#Pose a query to the DB
def queryDB(doc_type=''):
    #Database Settings
    user = dbSettings['user']
    password = dbSettings['password']
    db = dbSettings['db']
    host = dbSettings['host']
    port = dbSettings['port']
    
    #create query
    limitline= 'limit '+str(limitDocuments) if(limitDocuments!=-1) else ''
    
    if (doc_type=='web'):
        query = """
        select title, body
        from document
        where doc_type = 'web'
        """+limitline
    else:
        query = """
        (select body, doc_type
        from document
        where doc_type = 'web'
        """+limitline+"""
        )
        UNION
        (select body, doc_type
        from document
        where doc_type = 'twitter'
        """+limitline+"""
        ) """

    with warnings.catch_warnings():        
        #ignore warning
        warnings.simplefilter("ignore", category=sqlalchemy.exc.SAWarning)
        
        #connect with the help of the PostgreSQL URL
        url = 'postgresql://{}:{}@{}:{}/{}'
        url = url.format(user, password, host, port, db)
        con = sqlalchemy.create_engine(url, client_encoding='utf8')

        #results as pandas dataframe
        return pd.read_sql_query(query, con, index_col=None, coerce_float=True, params=None, parse_dates=None, chunksize=None)

#Pretty print of numbers (by https://stackoverflow.com/a/45846841)
def human_format(num):
    num = float('{:.3g}'.format(num))
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    return '{}{}'.format('{:f}'.format(num).rstrip('0').rstrip('.'), ['', 'K', 'M', 'B', 'T'][magnitude])