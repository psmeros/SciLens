import pandas as pd
import logging
import sys

#Removes duplicate links from a document
removeDuplicateLinks = True

#Ignores Tweets with  # of URLs greater than @urlLimit
applyUrlLimit = False
urlLimit = 10

#Timeout for URL resolving
urlTimout = 5

# Pandas Settings
pd.set_option('display.max_colwidth', 1024)


cachedDataFrame = 'cachedDataFrame.pkl'

gloveFile = '/home/psmeros/var/workspaces/nutrition-workspace/bigFiles/glove.6B.50d.txt' if sys.platform == 'linux' else \
            '/Users/smeros/workspace/etc/bigFiles/glove.6B/glove.6B.50d.txt'

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
