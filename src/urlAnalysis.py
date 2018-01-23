from settings import *
from utils import *

#Resolve all the URLs of the tweets
def resolveURLs():


    def resolve(url):

        resolved = {'resolvedURL':'', 'error':'', 'errorDesc':''}
        
        try:
            url = url[1:-1]
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

        return resolved

    spark = initSpark()

    documents = spark.read.option('sep', '\t').csv(twitter_urls, header=False, schema=StructType([StructField('id', StringType()), StructField('url', StringType())]))
    documents = documents.limit(limitDocuments) if(limitDocuments!=-1) else documents
    documents = documents.rdd

    documents = documents.flatMap(lambda s: Row(resolve(s.url)))
    documents = documents.filter(lambda s: s['error']=='NoError')

    documents = documents.map(lambda s : Row(url=s['resolvedURL']))

    documents.toDF().toPandas().to_csv('cache/'+sys._getframe().f_code.co_name+'.tsv', sep='\t')


#[DEPRECATED] Plot bar chart with the number of links per tweet
def plotCountLinks(limitDocuments=10):
    
    documents = queryDB(limitDocuments)
    documents = extractLinks(documents)
    
    #Limit for the bar charts
    urlLimit = 10
    
    #Writes the number of urls per document
    #If the number is ≥ @urlLimit then instead
    #of the real number, we write "≥ @urlLimit"
    documents['urls'] = documents['urls'].apply(lambda x: len(x) if len(x)<urlLimit else "≥"+str(urlLimit))
    
    #Groups by the doc_type (twitter, web) and the number of urls
    count =  documents.groupby(['doc_type','urls']).size()
    
    #Reformats the dataframe in order to create the plots
    docs = {}
    for doc_type in count.index.levels[0]:
        urls=[]
        for urlNum in count.index.levels[1]:
            urls.append(count.get((doc_type, urlNum),0))
        docs[doc_type]=urls

    #Slices the dataframe into 3 views.
    bothCount=pd.DataFrame(docs, index=count.index.levels[1])
    twitterCount=bothCount['twitter'][lambda x: x!=0]
    webCount=bothCount['web'][lambda x: x!=0]
 
    #Creates the 3 plots.
    plt.xticks(rotation=70)
    ax = twitterCount.plot.bar(fontsize=12, color='b', title='Twitter')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30)
    ax.set_xlabel("# urls")
    ax.set_ylabel("# documents")
    plt.show()
    
#[DEPRECATED] Plot pie chart with the various URL errors
def plotLinkErrors(limitDocuments=10):
    
    documents = resolveURLs(flattenLinks(extractLinks(queryDB(limitDocuments))))
    
    documents = documents[['doc_type','error']]
    
    #Groups and counts by error and plots the respective pies
    twitter = documents[documents['doc_type']=='twitter']
    numOfLinks = twitter.shape[0]
    ax = twitter.groupby('error').agg('count').apply(lambda x: x/numOfLinks).plot.pie(y='doc_type', autopct='%1.1f%%', title='Link Errors on Twitter', pctdistance=1.1, labeldistance=1.25)
    ax.set_ylabel('')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=6)
    web = documents[documents['doc_type']=='web']
    numOfLinks = web.shape[0]
    ax = web.groupby('error').agg('count').apply(lambda x: x/numOfLinks).plot.pie(y='doc_type', autopct='%1.1f%%', title='Link Errors on Web', pctdistance=1.1, labeldistance=1.25)
    ax.set_ylabel('')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=6)
    numOfLinks = documents.shape[0]
    ax = documents.groupby('error').agg('count').apply(lambda x: x/numOfLinks).plot.pie(y='doc_type', autopct='%1.1f%%', title='Link Errors on Both', pctdistance=1.1, labeldistance=1.25)
    ax.set_ylabel('')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=6)

    
#[DEPRECATED] Sort domains of the URLs by number of occurrences
def printDomains(limitDocuments=10):
    documents = resolveURLs(flattenLinks(extractLinks(queryDB(limitDocuments))))

    documents = documents[documents['error']=='NoError']
    documents = documents[['id','resolvedURL','doc_type']]

    documents['resolvedURL'] = documents['resolvedURL'].apply(lambda x: urlparse(x).hostname)

    print('Twitter')
    print(documents[documents['doc_type']=='twitter'][['id','resolvedURL']].groupby('resolvedURL').agg('count').sort_values('id', ascending=False))
    print('Web')
    print(documents[documents['doc_type']=='web'][['id','resolvedURL']].groupby('resolvedURL').agg('count').sort_values('id', ascending=False))
    print('Both')
    print(documents[['id','resolvedURL']].groupby('resolvedURL').agg('count').sort_values('id', ascending=False))

#[DEPRECATED] Get the DMOZ categories of a domain from Amazon Alexa (limited to 1000 requests/month)
def alexa():
    from myawis import CallAwis
    # Get credentials from: https://console.aws.amazon.com/iam/home#security_credential
    obj=CallAwis('www.ncbi.nlm.nih.gov','Categories','<Access Key ID>','<Secret Access Key>')
    print(obj.urlinfo())