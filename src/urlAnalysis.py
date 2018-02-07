from settings import *
from utils import *

#Resolve all the URLs of the tweets
def resolveURLs():

    def resolve(tweet_url, out_url):        
        try:
            #Follow the redirections of a url
            r = requests.head(out_url, allow_redirects=True, timeout=urlTimout)
            r.raise_for_status()
            return Row(tweet_url=tweet_url, out_url=r.url, out_error='NoError')

        #Catch the different errors       
        except requests.HTTPError as e:
            return Row(tweet_url=tweet_url, out_url=out_url, out_error='HTTPError')
        except:
            return Row(tweet_url=tweet_url, out_url=out_url, out_error='TimeoutError')


    spark = initSpark()

    documents = spark.read.option('sep', '\t').csv(twitterCorpusFile, header=False)
    documents = documents.limit(limitDocuments) if(limitDocuments!=-1) else documents
    documents = documents.rdd
    documents = documents.map(lambda s: Row(tweet_url=s[0], tweet=s[1], timestamp=s[2]))

    documents = documents.flatMap(lambda s: [Row(tweet_url=s.tweet_url, out_url=u) for u in re.findall(urlRegex, s.tweet) or ['']])


    documents = documents.map(lambda s: resolve(s.tweet_url, s.out_url))
    
    documents = documents.filter(lambda s: s.out_error=='NoError')

    documents.map(lambda s : Row(tweet_url=s.tweet_url, out_url=s.out_url)).toDF().write.csv('cache/'+sys._getframe().f_code.co_name, mode='overwrite')


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