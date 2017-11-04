import pandas as pd
import logging
import sys
import os


#Pickled dataframe
useCache = False
cachedDataFrame = 'cachedDataFrame.pkl'

#Use Spark for parallel processing
useSpark = False

#Limit retrieved documents
limitDocuments = 10

#Settings for database connection
dbSettings = {'user':'smeros', 'password':'vasoula', 'db':'sciArticles', 'host':'localhost', 'port':5432}

#GloVe Embeddings file
gloveFile = '/Users/smeros/workspace/etc/bigFiles/glove.6B/glove.6B.300d.txt'
#gloveFile = '/home/psmeros/var/workspaces/nutrition-workspace/bigFiles/glove.6B.300d.txt'
#gloveFile = '/home/smeros/glove_data/glove.6B.300d.txt'

if not os.path.exists(gloveFile):
    print(gloveFile,'embeddings not found')
    sys.exit(0)

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

#Plots directory
os.makedirs('plots', exist_ok=True)

#Pandas Settings
pd.set_option('display.max_colwidth', 1024)
