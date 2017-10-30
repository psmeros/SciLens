import pandas as pd
import logging
import sys
import os

#Pandas Settings
pd.set_option('display.max_colwidth', 1024)

#Pickled dataframe
cachedDataFrame = 'cachedDataFrame.pkl'
useCache = True

#Use Spark for parallel processing
useSpark = True


try: gloveFile = os.environ['gloveFile']
except: gloveFile = ('/home/psmeros/var/workspaces/nutrition-workspace/bigFiles/glove.6B.50d.txt' if sys.platform == 'linux' else '/Users/smeros/workspace/etc/bigFiles/glove.6B/glove.6B.50d.txt')

#Creates a query for the DB
def createQuery(limitDocuments, doc_type=''):
    limitline= 'limit '+str(limitDocuments) if(limitDocuments!=-1) else ''
    
    if (doc_type=='web'):
        query = """
        select title, body, topic_label
        from document, document_topic
        where doc_type = 'web' and id = document_id
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
    return query
