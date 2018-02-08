from settings import *
from utils import *


#Resolve all the URLs
def resolveURLs(tweet_url, out_url):        
    try:
        #Follow the redirections of a URL
        r = requests.head(out_url, allow_redirects=True, timeout=urlTimout)
        r.raise_for_status()
        return Row(tweet_url=tweet_url, out_url=r.url, out_error='')

    #Catch the different errors       
    except requests.HTTPError as e:
        return Row(tweet_url=tweet_url, out_url=out_url, out_error='HTTPError')
    except:
        return Row(tweet_url=tweet_url, out_url=out_url, out_error='TimeoutError')


#Create the first level of the diffusion graph
def first_level_graph():

    spark = initSpark()

    documents = spark.read.option('sep', '\t').csv(twitterCorpusFile, header=False)
    documents = documents.limit(limitDocuments) if(limitDocuments!=-1) else documents
    documents = documents.rdd


    documents = documents.map(lambda s: Row(tweet_url=s[0], tweet=s[1], timestamp=datetime.strptime(s[2], '%Y-%m-%d %H:%M:%S'), popularity=int(s[3]), RTs=int(s[4])))

    documents = documents.flatMap(lambda s: [Row(tweet_url=s.tweet_url, out_url=u) for u in re.findall(urlRegex, s.tweet) or ['']])


    documents = documents.map(lambda s: resolveURLs(s.tweet_url, s.out_url))
    
    documents = documents.filter(lambda s: s.out_error=='')

    documents.map(lambda s : Row(tweet_url=s.tweet_url, out_url=s.out_url)).toDF().write.csv('cache/'+sys._getframe().f_code.co_name, mode='overwrite')



#[DEPRECATED] Get the DMOZ categories of a domain from Amazon Alexa (limited to 1000 requests/month)
def alexa():
    from myawis import CallAwis
    # Get credentials from: https://console.aws.amazon.com/iam/home#security_credential
    obj=CallAwis('www.ncbi.nlm.nih.gov','Categories','<Access Key ID>','<Secret Access Key>')
    print(obj.urlinfo())