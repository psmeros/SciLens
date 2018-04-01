from settings import *
from utils import initSpark, rdd2tsv, analyze_url

institutions = pd.read_csv(institutionsFile, sep='\t')
institutions['URL'] = institutions['URL'].apply(lambda u: re.sub(r'^(www[0-9]?\.)|(web\.)', r'', u))
repositories = pd.read_csv(repositoriesFile)
repositories['URL'] = repositories['URL'].apply(lambda u: re.sub(r'^http://(www\.)?', r'', u))
blacklistURLs = open(blacklistURLsFile).read().splitlines()

url = 'http://sci-lens.github.io'
graph_nodes = {'tweetWithoutURL':url+'#tweetWithoutURL', 'HTTPError':url+'#HTTPError', 'TimeoutError':url+'#TimeoutError', 'TimeoutError':url+'#TimeoutError'}

#Resolve url
def resolveURL(url):
    if url=='':
        return graph_nodes['tweetWithoutURL']
        
    try:
        #Follow the redirections of a URL
        r = requests.head(url, allow_redirects='HEAD', timeout=urlTimout)
        if r.status_code != 403:
            r.raise_for_status()

        #Avoid blacklisted and flat URLs
        domain, path = analyze_url(r.url)
        if domain in blacklistURLs or path in ['', '/']:
            r.url = ''

        return r.url

    #Catch the different errors       
    except requests.HTTPError as e:
        return graph_nodes['HTTPError']
    except:
        return graph_nodes['TimeoutError']

#Create the first level of the diffusion graph from tweets
def graph_epoch_0():

    urlRegex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    spark = initSpark()

    documents = spark.sparkContext.textFile(twitterCorpusFile, minPartitions=(conf['cores']-1)) 

    documents = documents.map(lambda r: (lambda l=r.split('\t'): Row(source_url=l[0], tweet=l[1], timestamp=datetime.strptime(l[2], '%Y-%m-%d %H:%M:%S'), popularity=int(l[3]), RTs=int(l[4]), user_country=l[5]))())

    documents = documents.flatMap(lambda r: [Row(source_url=r.source_url, timestamp=r.timestamp, popularity=r.popularity, RTs=r.RTs, user_country=r.user_country, target_url=resolveURL(u)) for u in re.findall(urlRegex, r.tweet) or ['']])

    documents = documents.map(lambda r : '\t'.join(str(a) for a in [r.source_url, r.timestamp, r.popularity, r.RTs, r.user_country, r.target_url]))

    rdd2tsv(documents, diffusion_graph_dir+'epoch_0.tsv', ['source_url','timestamp', 'popularity', 'RTs', 'user_country', 'target_url'])

#Plot URL decay per year
def plot_URL_decay():
    df = pd.read_csv(diffusion_graph_dir+'epoch_0.tsv', sep='\t')
    df['date'] = df['timestamp'].apply(lambda s : datetime.strptime(s, '%Y-%m-%d %H:%M:%S').year)
    df['target_url'] = df['target_url'].apply(lambda u: u if u in [graph_nodes['tweetWithoutURL'], graph_nodes['HTTPError'], graph_nodes['TimeoutError']] else 'working URL')
    df['Tweets with'] = df['target_url'].map(lambda n: 'HTTP error in outgoing URL' if n == graph_nodes['HTTPError'] else 'timeout error in outgoing URL' if n == graph_nodes['TimeoutError'] else 'no URL' if n == graph_nodes['tweetWithoutURL'] else 'working URL')
    ax = df[['source_url', 'date','Tweets with']].pivot_table(index='date', columns='Tweets with',aggfunc='count').T.reset_index(level=0, drop=True).T.fillna(1).plot(logy=True, figsize=(10,10), sort_columns=True)

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
    
    documents = spark.sparkContext.parallelize(frontier, numSlices=(conf['cores']-1))

    documents = documents.flatMap(lambda r: [Row(source_url=r, target_url=l) for l in get_out_links(r) or ['']])

    documents = documents.map(lambda r : '\t'.join(str(a) for a in [r.source_url, r.target_url]))
    
    rdd2tsv(documents, file, ['source_url', 'target_url'])


#Create diffusion graph
def create_graph():

    if not useCache or not os.path.exists(diffusion_graph_dir+'epoch_0.tsv'):
        graph_epoch_0()

    df = pd.read_csv(diffusion_graph_dir+'epoch_0.tsv', sep='\t').dropna()    
    tweets = df.copy().drop('target_url', axis=1).drop_duplicates('source_url')
    #beutify country names
    tweets = tweets.merge(pd.read_csv(countriesFile).rename(columns={'Name':'Country'}), left_on='user_country', right_on='Code').drop(['user_country', 'Code'], axis=1).set_index('source_url')
    tweets.loc[tweets['Country'] == 'United States', 'Country'] = 'USA'
    print('Initial Tweets:', len(tweets))

    epochs = 1
    G=nx.DiGraph()
    for epoch in range(0, epochs):

        df = pd.read_csv(diffusion_graph_dir+'epoch_'+str(epoch)+'.tsv', sep='\t').dropna()
        G =  nx.compose(G, nx.from_pandas_edgelist(df, source='source_url', target='target_url', create_using=nx.DiGraph()))
        frontier = [x for x in G.nodes() if G.out_degree(x) == 0]

        if not useCache or not os.path.exists(diffusion_graph_dir+'epoch_'+str(epoch+1)+'.tsv'):
            graph_epoch_n(frontier, diffusion_graph_dir+'epoch_'+str(epoch+1)+'.tsv')

        print('Connected Components:', nx.number_connected_components(G.to_undirected()))
        print('Frontier Size:', len(frontier))
    

    print(G.in_degree(graph_nodes['tweetWithoutURL']))
    print(G.in_degree(graph_nodes['HTTPError']))
