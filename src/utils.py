from urllib.request import urlopen
from urllib.parse import urlparse
from urllib.error import HTTPError, URLError
from ssl import CertificateError
from socket import timeout as SocketTimeoutError
import re
import sys
import os
import numpy as np


from settings import *

#Loads the Glove Embeddings file
def loadGloveEmbeddings(filename):
    print('Loading', filename,'embeddings file')
    if not os.path.exists(filename):
        print(filename,'embeddings not found')
        return None
    words = {} #key= word, value=embeddings
    with open(filename, "r") as f:
        for line in f:
            tokens = line.strip().split()
            words[tokens[0]] = np.array([float(x) for x in tokens[1:]])
    return words, len(next(iter(words.values())))    


#Resolves all the URLs of the documents.
def resolveURLs(documents):

    #Resolves a URL.
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


#Flattens documents with multiple URLs.
def flattenLinks(documents):
    
    #Filters out documents that have more than @urlLimit.
    if (applyUrlLimit):
        documents = documents[documents['urls'].apply(lambda x: len(x)) <urlLimit]
    
    #Converts pairs of <id, [url1, url2, ...]> to <id, url1>, <id, url2>.
    documents = pd.DataFrame([(row[0], link, row[2]) for row in documents.itertuples() for link in row[1]], columns=['id', 'url', 'doc_type'])
    
    #Removes duplicates.
    if (removeDuplicateLinks):
        documents = documents.drop_duplicates()
    return documents


#Extracts URLs using regex.
def extractLinks(documents):

    urlRegex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    
    documents['body'] = documents['body'].apply(lambda x: re.findall(urlRegex, x)).to_frame()
    documents.rename(columns={'body': 'urls'}, inplace=True)
    return documents


#Poses query to DB.
def queryDB(query, user='smeros', password='', db='sciArticles', host='localhost', port=5432):
    import sqlalchemy
    import warnings

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

# Show progress bar.
# from ipywidgets import FloatProgress
# from IPython.display import display
# f = FloatProgress(min=0, max=documents.shape[0])
# display(f)
# f.value += 1
    