from settings import *
from utils import initSpark


#Resolve url
def resolveURL(url):
    if url=='':
        return 'http://TweetWithoutURL.org'
        
    try:
        #Follow the redirections of a URL
        r = requests.head(url, allow_redirects=True, timeout=urlTimout)
        r.raise_for_status()
        return r.url

    #Catch the different errors       
    except requests.HTTPError as e:
        return 'http://HTTPError.org'
    except:
        return 'http://TimeoutError.org'


#Create the first level of the diffusion graph
def first_level_graph():

    spark = initSpark()

    documents = spark.sparkContext.textFile(twitterCorpusFile) 

    documents = documents.map(lambda r: (lambda l=r.split('\t'): Row(tweet_url=l[0], tweet=l[1], timestamp=datetime.strptime(l[2], '%Y-%m-%d %H:%M:%S'), popularity=int(l[3]), RTs=int(l[4])))())

    documents = documents.flatMap(lambda r: [Row(tweet_url=r.tweet_url, timestamp=r.timestamp, popularity=r.popularity, RTs=r.RTs, out_url=u) for u in re.findall(urlRegex, r.tweet) or ['']])

    documents = documents.map(lambda r: Row(tweet_url=r.tweet_url, timestamp=r.timestamp, popularity=r.popularity, RTs=r.RTs, out_url=resolveURL(r.out_url)))

    documents.map(lambda r : '\t'.join(str(a) for a in [r.tweet_url, r.timestamp, r.popularity, r.RTs, r.out_url])).saveAsTextFile('cache/'+sys._getframe().f_code.co_name)
