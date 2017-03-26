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
pd.set_option('display.width', 1024)
cachedDataFrame = 'cachedDataFrame.pkl'

#Creates the query for the DB
def createQuery(limitDocuments):
    limitline= 'limit '+str(limitDocuments) if(limitDocuments!=-1) else ''
    
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