from settings import *
from utils import initSpark, rdd2tsv, analyze_url

#Resolve url
def resolveURL(url):
    if url=='':
        return 'http://TweetWithoutURL.org'
        
    try:
        #Follow the redirections of a URL
        r = requests.head(url, allow_redirects='HEAD', timeout=urlTimout)
        if r.status_code != 403:
            r.raise_for_status()

        #Avoid blacklisted and flat URLs
        domain, path = analyze_url(r.url)
        if domain in blacklistURLs or path == '':
            r.url = ''

        return r.url

    #Catch the different errors       
    except requests.HTTPError as e:
        return 'http://HTTPError.org'
    except:
        return 'http://TimeoutError.org'

#Create the first level of the diffusion graph
def first_level_graph():

    spark = initSpark()

    documents = spark.sparkContext.textFile(twitterCorpusFile, minPartitions=(cores-1)) 

    documents = documents.map(lambda r: (lambda l=r.split('\t'): Row(tweet_url=l[0], tweet=l[1], timestamp=datetime.strptime(l[2], '%Y-%m-%d %H:%M:%S'), popularity=int(l[3]), RTs=int(l[4]), user_country=l[5]))())

    documents = documents.flatMap(lambda r: [Row(tweet_url=r.tweet_url, timestamp=r.timestamp, popularity=r.popularity, RTs=r.RTs, user_country=r.user_country, out_url=resolveURL(u)) for u in re.findall(urlRegex, r.tweet) or ['']])

    documents = documents.map(lambda r : '\t'.join(str(a) for a in [r.tweet_url, r.timestamp, r.popularity, r.RTs, r.user_country, r.out_url]))

    rdd2tsv(documents, first_level_graph_file, ['tweet_url','timestamp', 'popularity', 'RTs', 'user_country', 'out_url'])

#Plot URL decay per year
def plot_URL_decay():
    df = pd.read_csv(first_level_graph_file, sep='\t')
    df['date'] = df['timestamp'].apply(lambda s : datetime.strptime(s, '%Y-%m-%d %H:%M:%S').year)
    df['out_url'] = df['out_url'].apply(lambda u: u if u in ['http://TweetWithoutURL.org', 'http://HTTPError.org', 'http://TimeoutError.org'] else 'http://WorkingURL.org')
    df['Tweets with'] = df['out_url'].map(lambda n: 'HTTP error in outgoing URL' if n == 'http://HTTPError.org' else 'timeout error in outgoing URL' if n == 'http://TimeoutError.org' else 'no URL' if n == 'http://TweetWithoutURL.org' else 'working URL')
    ax = df[['tweet_url', 'date','Tweets with']].pivot_table(index='date', columns='Tweets with',aggfunc='count').T.reset_index(level=0, drop=True).T.fillna(1).plot(logy=True, figsize=(10,10), sort_columns=True)

#Get outgoing links from article
def get_out_links(url):
    links = []
    try:
        soup = BeautifulSoup(urlopen(url, timeout=urlTimout), 'html.parser')
    except:
        return links

    url = get_url_domain(url)
    for link in soup.findAll('a'):
        u = get_url_domain(link.get('href') or '')
        if (url not in u) and (u not in url) and (u not in ['']+blacklistURLs):
            links.append(link.get('href'))

    return list(set(links))

#Create the second level of the diffusion graph
def second_level_graph():

    spark = initSpark()
    
    documents = spark.sparkContext.textFile(second_level_urls_file, minPartitions=(cores-1))

    documents = documents.map(lambda r: (lambda l=r.split('\t'): Row(url=l[0]))())

    documents = documents.flatMap(lambda r: [Row(url=r.url, out_url=l) for l in get_out_links(r.url) or ['']])
    
    documents.map(lambda r : '\t'.join(str(a) for a in [r.url, r.out_url])).saveAsTextFile(second_level_graph_file)


#Create diffusion graph
def create_graph(): 
    if not useCache or not os.path.exists(first_level_graph_file):
        first_level_graph()

    df = pd.read_csv(first_level_graph_file, sep='\t')

    G = nx.from_pandas_edgelist(df, source='tweet_url', target='out_url', create_using=nx.DiGraph())

    for attr in ['timestamp', 'popularity', 'RTs', 'user_country']:
        nx.set_node_attributes(G, df[attr].to_dict(), attr)


    print(nx.number_connected_components(G.to_undirected()))
    print(G.in_degree('http://TweetWithoutURL.org'))
    print(G.in_degree('http://HTTPError.org'))
    for x in G.nodes(): 
        if G.out_degree(x) == 0:
            print(x)
