from settings import *
from utils import initSpark, rdd2tsv, analyze_url, get_graph_filename

institutions = pd.read_csv(institutionsFile, sep='\t')
institutions['URL'] = institutions['URL'].apply(lambda u: re.sub(r'^(www[0-9]?\.)|(web\.)', r'', u))


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

#Create the first level of the diffusion graph from tweets
def graph_epoch_0_twitter():

    spark = initSpark()

    documents = spark.sparkContext.textFile(twitterCorpusFile, minPartitions=(cores-1)) 

    documents = documents.map(lambda r: (lambda l=r.split('\t'): Row(source_url=l[0], tweet=l[1], timestamp=datetime.strptime(l[2], '%Y-%m-%d %H:%M:%S'), popularity=int(l[3]), RTs=int(l[4]), user_country=l[5]))())

    documents = documents.flatMap(lambda r: [Row(source_url=r.source_url, timestamp=r.timestamp, popularity=r.popularity, RTs=r.RTs, user_country=r.user_country, target_url=resolveURL(u)) for u in re.findall(urlRegex, r.tweet) or ['']])

    documents = documents.map(lambda r : '\t'.join(str(a) for a in [r.source_url, r.timestamp, r.popularity, r.RTs, r.user_country, r.target_url]))

    rdd2tsv(documents, get_graph_filename(0), ['source_url','timestamp', 'popularity', 'RTs', 'user_country', 'target_url'])

#Plot URL decay per year
def plot_URL_decay():
    df = pd.read_csv(get_graph_filename(0), sep='\t')
    df['date'] = df['timestamp'].apply(lambda s : datetime.strptime(s, '%Y-%m-%d %H:%M:%S').year)
    df['out_url'] = df['out_url'].apply(lambda u: u if u in ['http://TweetWithoutURL.org', 'http://HTTPError.org', 'http://TimeoutError.org'] else 'http://WorkingURL.org')
    df['Tweets with'] = df['out_url'].map(lambda n: 'HTTP error in outgoing URL' if n == 'http://HTTPError.org' else 'timeout error in outgoing URL' if n == 'http://TimeoutError.org' else 'no URL' if n == 'http://TweetWithoutURL.org' else 'working URL')
    ax = df[['tweet_url', 'date','Tweets with']].pivot_table(index='date', columns='Tweets with',aggfunc='count').T.reset_index(level=0, drop=True).T.fillna(1).plot(logy=True, figsize=(10,10), sort_columns=True)

#Get outgoing links from article
def get_out_links(url):
    links = []
    # try:
    #     soup = BeautifulSoup(urlopen(url, timeout=urlTimout), 'html.parser')
    # except:
    #     return links

    # url = get_url_domain(url)
    # for link in soup.findAll('a'):
    #     u = get_url_domain(link.get('href') or '')
    #     if (url not in u) and (u not in url) and (u not in ['']+blacklistURLs):
    #         links.append(link.get('href'))

    return list(set(links))

#Create the nth level of the diffusion graph
def graph_epoch_n(frontier, file):

    spark = initSpark()
    
    documents = spark.sparkContext.parallelize(frontier, numSlices=(cores-1))

    documents = documents.flatMap(lambda r: [Row(source_url=r, target_url=l) for l in get_out_links(r) or ['']])

    documents = documents.map(lambda r : '\t'.join(str(a) for a in [r.source_url, r.target_url]))
    
    rdd2tsv(documents, file, ['source_url', 'target_url'])


#Create diffusion graph
def create_graph():

    if not useCache or not os.path.exists(get_graph_filename(0)):
        graph_epoch_0_twitter()

    df = pd.read_csv(get_graph_filename(0), sep='\t').dropna()    
    tweets = df.copy().drop('target_url', axis=1).drop_duplicates('source_url')
    #beutify country names
    tweets = tweets.merge(pd.read_csv(countriesFile).rename(columns={'Name':'Country'}), left_on='user_country', right_on='Code').drop(['user_country', 'Code'], axis=1).set_index('source_url')
    tweets.loc[tweets['Country'] == 'United States', 'Country'] = 'USA'
    print('Initial Tweets:', len(tweets))

    epochs = 2
    G=nx.DiGraph()
    for epoch in range(0, epochs):

        df = pd.read_csv(get_graph_filename(epoch), sep='\t').dropna()
        G =  nx.compose(G, nx.from_pandas_edgelist(df, source='source_url', target='target_url', create_using=nx.DiGraph()))
        frontier = [x for x in G.nodes() if G.out_degree(x) == 0]

        if not useCache or not os.path.exists(get_graph_filename(epoch+1)):
            graph_epoch_n(frontier, get_graph_filename(epoch+1))

        print('Connected Components:', nx.number_connected_components(G.to_undirected()))
        print('Frontier Size:', len(frontier))
    

    # print(G.in_degree('http://TweetWithoutURL.org'))
    # print(G.in_degree('http://HTTPError.org'))
