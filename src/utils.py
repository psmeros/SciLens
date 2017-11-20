from urllib.request import urlopen
from urllib.parse import urlparse
from urllib.error import HTTPError, URLError
from ssl import CertificateError
from socket import timeout as SocketTimeoutError
import sqlalchemy
import warnings
import re
import sys
import os.path
from time import time
import numpy as np
from sklearn.neighbors import NearestNeighbors
from spacy.en import English

from settings import *
from gloveEmbeddings import *

# Create Keyword Lists
nlp = English()
authorityKeywords = [nlp(x)[0].lemma_ for x in ['expert', 'scientist', 'researcher', 'professor', 'author', 'paper', 'report', 'study', 'analysis', 'research', 'survey', 'release']]
empiricalKeywords = [nlp(x)[0].lemma_ for x in ['study', 'people']]
actionsKeywords = [nlp(x)[0].lemma_ for x in ['prove', 'demonstrate', 'reveal', 'state', 'mention', 'report', 'say', 'show', 'announce', 'claim', 'suggest', 'argue', 'predict', 'believe', 'think']]


#Reads/Writes the results of *func* from/to cache.
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

#Poses a query to the DB
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
        '''Returns a connection and a metadata object'''
        
        #ignore warning
        warnings.simplefilter("ignore", category=sqlalchemy.exc.SAWarning)
        
        # We connect with the help of the PostgreSQL URL
        url = 'postgresql://{}:{}@{}:{}/{}'
        url = url.format(user, password, host, port, db)

        # The return value of create_engine() is our connection object
        con = sqlalchemy.create_engine(url, client_encoding='utf8')

        # We then bind the connection to MetaData()
        #meta = sqlalchemy.MetaData(bind=con, reflect=True)

        #results as pandas dataframe
        results = pd.read_sql_query(query, con, index_col=None, coerce_float=True, params=None, parse_dates=None, chunksize=None)
        
        return results

#Pretty print of numbers
def human_format(num):
    #by https://stackoverflow.com/a/45846841  
    num = float('{:.3g}'.format(num))
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    return '{}{}'.format('{:f}'.format(num).rstrip('0').rstrip('.'), ['', 'K', 'M', 'B', 'T'][magnitude])


##URL Handling##

#Resolves all the URLs of the documents
def resolveURLs(documents):
    #Timeout for URL resolving
    urlTimout = 5

    def resolve(url):
        
        resolved = {'resolvedURL':'', 'error':'', 'errorDesc':''}
        
        try:
            #Follows the redirections of a url.
            resolvedURL = urlopen(url,timeout=urlTimout).geturl()

            #Some *.ly link do not redirect but neither return an error code.
            if(urlparse(resolvedURL).geturl().endswith('.ly')):
                resolved = {'resolvedURL':resolvedURL, 'error':'NoRedirectError', 'errorDesc':''}
            else:
                resolved = {'resolvedURL':resolvedURL, 'error':'NoError', 'errorDesc':''}

        #Catches the different errors.        
        except HTTPError as e:
            resolved = {'resolvedURL':url, 'error':'HTTPError', 'errorDesc':str(e.code)}
        except URLError as e:
            resolved = {'resolvedURL':url, 'error':'URLError', 'errorDesc':str(e)}
        except ConnectionResetError as e:
            resolved = {'resolvedURL':url, 'error':'ConnectionResetError', 'errorDesc':str(e)}
        except CertificateError as e:
            resolved = {'resolvedURL':url, 'error':'CertificateError', 'errorDesc':str(e)}
        except SocketTimeoutError as e:
            resolved = {'resolvedURL':url, 'error':'SocketTimeoutError', 'errorDesc':str(e)}
        except UnicodeEncodeError as e:    
            resolved = {'resolvedURL':url, 'error':'UnicodeEncodeError', 'errorDesc':str(e)}            
        except:
            with open('error.log', 'a') as f: f.write('URL: '+url+' Error: '+str(sys.exc_info())+'\n')

        return pd.Series(resolved)
    
    documents = pd.concat((documents, documents['url'].apply(lambda x: resolve(x))), axis=1)
    return documents

#Flattens documents with multiple URLs
def flattenLinks(documents):
    applyUrlLimit = False
    urlLimit = 10
    removeDuplicateLinks = True

    #Ignores Tweets with #URLs greater than @urlLimit
    if (applyUrlLimit):
        documents = documents[documents['urls'].apply(lambda x: len(x)) <urlLimit]
    
    #Converts pairs of <id, [url1, url2, ...]> to <id, url1>, <id, url2>
    documents = pd.DataFrame([(row[0], link, row[2]) for row in documents.itertuples() for link in row[1]], columns=['id', 'url', 'doc_type'])
    
    #Removes duplicates
    if (removeDuplicateLinks):
        documents = documents.drop_duplicates()
    return documents


#Extracts URLs using regex
def extractLinks(documents):

    urlRegex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    
    documents['body'] = documents['body'].apply(lambda x: re.findall(urlRegex, x)).to_frame()
    documents.rename(columns={'body': 'urls'}, inplace=True)
    return documents



##Text Summarization##

#Converts the title and each sentence of the body of a document
#to vectors and then runs a NN algorithm
def titleAndBodyNeighbors(limitDocuments=10):
    numOfNeighbors = 3
    
    query = createQuery(limitDocuments, 'web')
    documents = queryDB(query)    
     
    documents['bodyVecs'] = documents['body'].apply(lambda b: body2Vec(b))
    documents['titleVec'] = documents['title'].apply(lambda t: title2Vec(t))
    
    def nn(titleVec, bodyVecs, body):    
        nbrs = NearestNeighbors(n_neighbors=numOfNeighbors, algorithm='ball_tree').fit(np.array(bodyVecs))
        distances, indices = nbrs.kneighbors(titleVec.reshape(1, -1))
        
        allSentences = [sent.text.strip() for sent in nlp(body).sents]
        sentences = []
        for i in indices.tolist()[0]:
            sentences.append(allSentences[int(i)])
        return sentences
    
    documents['nn'] = documents.apply(lambda d: nn(d['titleVec'],d['bodyVecs'], d['body']), axis=1)

    documents = documents.drop(['bodyVecs', 'titleVec'], axis=1)
    return documents

#Creates a summary of a document based on gensim model (textRank)
def summarizeDocument(document):
    from gensim.summarization.summarizer import summarize       
    summary = summarize(document)
    print (document, "-----", summary)