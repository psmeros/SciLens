import pandas as pd
import logging

#Removes duplicate links from a document
removeDuplicateLinks = True

#Ignores Tweets with  # of URLs greater than @urlLimit
applyUrlLimit = False
urlLimit = 10

#Timeout for URL resolving
urlTimout = 5

# Pandas Settings
pd.set_option('display.max_colwidth', 2048)

cachedDataFrame = 'cachedDataFrame.pkl'

gloveFile = '/Users/smeros/workspace/etc/bigFiles/glove.6B/glove.6B.50d.txt'

#Creates a query for the DB
def createQuery(limitDocuments, doc_type=''):
    limitline= 'limit '+str(limitDocuments) if(limitDocuments!=-1) else ''
    
    if (doc_type=='web'):
        query = """
        select title, body 
        from document
        where doc_type = 'web'
        """+limitline
    else:
        query = """
        (select id, body, doc_type
        from document
        where doc_type = 'web'
        """+limitline+"""
        )
        UNION
        (select id, body, doc_type
        from document
        where doc_type = 'twitter'
        """+limitline+"""
        ) """
    return query
