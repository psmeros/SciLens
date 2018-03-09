from settings import *
from utils import initSpark
from scraping import get_out_links


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

    documents = documents.map(lambda r: (lambda l=r.split('\t'): Row(tweet_url=l[0], tweet=l[1], timestamp=datetime.strptime(l[2], '%Y-%m-%d %H:%M:%S'), popularity=int(l[3]), RTs=int(l[4]), user_country=l[5]))())

    documents = documents.flatMap(lambda r: [Row(tweet_url=r.tweet_url, timestamp=r.timestamp, popularity=r.popularity, RTs=r.RTs, user_country=r.user_country, out_url=u) for u in re.findall(urlRegex, r.tweet) or ['']])

    documents = documents.map(lambda r: Row(tweet_url=r.tweet_url, timestamp=r.timestamp, popularity=r.popularity, RTs=r.RTs, user_country=r.user_country, out_url=resolveURL(r.out_url)))

    documents.map(lambda r : '\t'.join(str(a) for a in [r.tweet_url, r.timestamp, r.popularity, r.RTs, r.user_country, r.out_url])).saveAsTextFile(first_level_graph_file)


#Create the second level of the diffusion graph
def second_level_graph():

    spark = initSpark()

    blacklist = [x.strip('\n') for x in open(blacklistFile).readlines()]

    documents = spark.sparkContext.textFile(second_level_urls_file)
    documents = documents.map(lambda r: (lambda l=r.split('\t'): Row(url=l[0]))())

    documents = documents.flatMap(lambda r: [Row(url=r.url, out_url=l) for l in get_out_links(r.url, blacklist) or ['']])
    
    documents.map(lambda r : '\t'.join(str(a) for a in [r.url, r.out_url])).saveAsTextFile(second_level_graph_file)