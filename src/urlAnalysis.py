from urllib.request import urlopen
from urllib.parse import urlparse
from urllib.error import HTTPError, URLError
from ssl import CertificateError
from socket import timeout as SocketTimeoutError
import re
import sys

from settings import *


#Resolve all the URLs of the documents
def resolveURLs(documents):
    #Timeout for URL resolving
    urlTimout = 5

    def resolve(url):
        
        resolved = {'resolvedURL':'', 'error':'', 'errorDesc':''}
        
        try:
            #Follow the redirections of a url
            resolvedURL = urlopen(url,timeout=urlTimout).geturl()

            #Some *.ly link do not redirect but neither return an error code
            if(urlparse(resolvedURL).geturl().endswith('.ly')):
                resolved = {'resolvedURL':resolvedURL, 'error':'NoRedirectError', 'errorDesc':''}
            else:
                resolved = {'resolvedURL':resolvedURL, 'error':'NoError', 'errorDesc':''}

        #Catch the different errors       
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

#Flatten documents with multiple URLs
def flattenLinks(documents):
    applyUrlLimit = False
    urlLimit = 10
    removeDuplicateLinks = True

    #Ignore Tweets with #URLs greater than @urlLimit
    if (applyUrlLimit):
        documents = documents[documents['urls'].apply(lambda x: len(x)) <urlLimit]
    
    #Convert pairs of <id, [url1, url2, ...]> to <id, url1>, <id, url2>
    documents = pd.DataFrame([(row[0], link, row[2]) for row in documents.itertuples() for link in row[1]], columns=['id', 'url', 'doc_type'])
    
    #Remove duplicates
    if (removeDuplicateLinks):
        documents = documents.drop_duplicates()
    return documents


#Extract URLs using regex
def extractLinks(documents):

    urlRegex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    
    documents['body'] = documents['body'].apply(lambda x: re.findall(urlRegex, x)).to_frame()
    documents.rename(columns={'body': 'urls'}, inplace=True)
    return documents