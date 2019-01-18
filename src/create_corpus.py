import os
import random
import re
from time import time, sleep
from datetime import datetime
from math import ceil, exp, floor
from urllib.parse import urlsplit
from urllib.request import urlopen

import networkx as nx
import pandas as pd
import requests
from bs4 import BeautifulSoup
from newspaper import Article
from pyspark import SparkConf
from pyspark.sql import Row, SparkSession

from settings import scilens_dir

############################### CONSTANTS ###############################

#Spark conf
conf = {'memory':8, 'cores':4, 'partitions':4*20}
#conf = {'memory':64, 'cores':24, 'partitions':24*20}
#conf = {'memory':252, 'cores':48, 'partitions':48*20}

#File with institutions metadata
institutions = pd.read_csv(scilens_dir + 'small_files/institutions/metadata.tsv', sep='\t')
institutions['URL'] = institutions['URL'].apply(lambda u: re.sub(r'^(www[0-9]?\.)|(web\.)', r'', u))

#File with academic repositories
repositories = pd.read_csv(scilens_dir + 'small_files/repositories/academic_repositories.csv')
repositories['URL'] = repositories['URL'].apply(lambda u: re.sub(r'^http://(www\.)?', r'', u))
sources = institutions['URL'].tolist() + repositories['URL'].tolist()

#Blacklisted URLs
blacklistURLs = open(scilens_dir + 'small_files/blacklist/urls.txt').read().splitlines()

diffusion_graph_dir = scilens_dir + 'cache/test_diffusion_graph/'

#Diffusion graph nodes
project_url = 'http://sci-lens.org'
graph_nodes = {'tweetWithoutURL':project_url+'#tweetWithoutURL', 'HTTPError':project_url+'#HTTPError', 'TimeoutError':project_url+'#TimeoutError', 'institution':project_url+'#institution', 'repository':project_url+'#repository', 'source':project_url+'#source'}

#maximum outgoing links from a webpage
max_outgoing_links = 10

#URL redirection settings
url_timeout = 1

#Components ratio for graph construction
components_ratio = 0.10

############################### ######### ###############################


################################ HELPERS ################################

#Scrap CWUR World University Rankings
def scrap_cwur(institutions_file):
	for year in ['2017']:

	    soup = BeautifulSoup(urlopen('http://cwur.org/'+year+'.php'), 'html.parser')
	    table = soup.find('table', attrs={'class' : 'table'})

	    headers = ['URL']+[header.text for header in table.find_all('th')]+['Year']

	    rows = []

	    for row in table.find_all('tr')[1:]:
	        soup = BeautifulSoup(urlopen('http://cwur.org'+row.find('a')['href'][2:]), 'html.parser')
	        url = soup.find('table', attrs={'class' : 'table table-bordered table-hover'}).find_all('td')[-1].text
	        rows.append([url]+[val.text for val in row.find_all('td')]+[year])

	    df = pd.DataFrame(rows, columns = headers)
	    df = df.applymap(lambda x: x.strip('+')).drop('World Rank', axis=1).reset_index().rename(columns={'index':'World Rank'})

	    df.to_csv(institutions_file, sep='\t', index=False)


#Scrap nutritionfacts.org topics
def scrap_nutritionfacts(vocabulary_file):
    soup = BeautifulSoup(urlopen('https://nutritionfacts.org/topics'), 'html.parser')
    div = soup.find('div', attrs={'class' : 'topics-index'})

    with open(vocabulary_file, 'w') as f:
    	for t in div.find_all('a', title=True):
    		f.write(t['title'] + '\n')

#Scrap replies from twitter (without using the API)
def scrap_twitter_replies(url, sleep_time):
    try:
        soup = BeautifulSoup(urlopen(url), 'html.parser')
    except:
        return []
        
    sleep(sleep_time)
    replies = []
    for d in soup.find_all('div', attrs={'class' : 'js-tweet-text-container'}):
        try:
            replies.append(d.find('p', attrs={'class':"TweetTextSize js-tweet-text tweet-text", 'data-aria-label-part':'0', 'lang':'en'}).get_text())
        except:
            continue

    return replies

#Find the domain and the path of an http url
def analyze_url(url):
    try:
        url=urlsplit(url)
        domain = re.sub(r'^(http(s)?://)?(www\.)?', r'', url.netloc)
        path = '' if domain == '' else url.path
        return domain, path
    except:
        return url, ''

#Compare two domains
def same_domains(domain_1, domain_2):
    if domain_1.count('.') == 2:
        domain_1 = ('.').join(domain_1.split('.')[1:])
    if domain_2.count('.') == 2:
        domain_2 = ('.').join(domain_2.split('.')[1:])
    
    if domain_1 in domain_2 or domain_2 in domain_1:
        return True
    return False

#scrap html page as a browser
def get_html(url):
	headers = {"User-Agent":"Mozilla/5.0 (X11; U; Linux i686) Gecko/20071127 Firefox/2.0.0.11"}
	r = requests.get(url, allow_redirects='HEAD', timeout=url_timeout, headers=headers)
	return BeautifulSoup(r.content, 'html.parser', from_encoding="iso-8859-1").find('body')


#Spark setup
def init_spark():
    spark_conf = SparkConf()
    spark_conf.setAppName('diffusion graph')
    spark_conf.setMaster('local['+str(conf['partitions'])+']')
    spark_conf.set('spark.executor.memory', str(floor(conf['memory']*0.9))+'G')
    spark_conf.set('spark.driver.memory', str(ceil(conf['memory']*0.1))+'G')
    spark_conf.set('spark.hadoop.validateOutputSpecs', 'false')
    spark = SparkSession.builder.config(conf=spark_conf).getOrCreate()
    spark.sparkContext.setLogLevel('ERROR')
    return spark.sparkContext

#Write RDD to TSV file (with header)
def rdd2tsv(rdd, file, attributes):
    rdd.saveAsTextFile(file+'_files')
    os.system('echo "' + '\t'.join(attributes) + '" > ' + file + '; cat ' + file + '_files/* >> ' + file + '; rm -r ' + file + '_files')

#Read diffusion graph
def read_graph(graph_file):
    G = nx.DiGraph()
    edges = open(graph_file).read().splitlines()
    for e in edges:
        [e0, e1] = e.split('\t')
        G.add_edge(e0, e1)
    return G

def write_graph(G, graph_file):
    with open(graph_file, 'w') as f:
        for edge in G.edges:
                f.write(edge[0] + '\t' + edge[1] + '\n')
################################ ####### ################################

#Resolve short url
def resolve_short_url(url):
    if url=='':
        return graph_nodes['tweetWithoutURL']
        
    try:
        #Follow the redirections of a URL
        r = requests.head(url, allow_redirects='HEAD', timeout=url_timeout)
        if r.status_code != 403:            
            r.raise_for_status()

        #Avoid blacklisted and flat URLs
        domain, path = analyze_url(r.url)
        if domain in blacklistURLs or path in ['', '/']:
            r.url = ''

        return re.sub('\?.*', '', re.sub('^http://', 'https://', r.url))

    #Catch the different errors       
    except requests.HTTPError as e:
        return graph_nodes['HTTPError']
    except:
        return graph_nodes['TimeoutError']

#Get outgoing links from article
def get_out_links(url, epoch_decay, last_pass):
    
    #custom urls for special nodes
    if url.startswith(project_url):
        return ['']

    domain, path = analyze_url(url)

    #predefined sources
    for s in sources:
        if s in domain:
            return [s]

    #other sources
    if any(suffix in domain for suffix in ['.edu', '.ac.uk', '.gov']):
        if path in ['', '/']:
            return[graph_nodes['repository']]
        else:
            return[domain]

    #Do not expand links over the last pass
    if last_pass:
        return ['']

    try:
        html = get_html(url)
        if not html:
            return ['']    
    except:
        return ['']

    #get all links except for self and blacklisted links
    links = []
    source_links = []
    for link in html.findAll('a'):
        link = link.get('href') or ''
        link_domain, link_path = analyze_url(link)
        if not same_domains(domain, link_domain) and link_domain not in blacklistURLs and link_path not in ['', '/'] and len(links) < max_outgoing_links:
            link = re.sub('\?.*', '', re.sub('^http://', 'https://', link))
            links.append(link)
            for s in sources:
                if s in link_domain:
                    source_links.append(link)
                    break

    #if there are links to the predefined sources, return only them
    if source_links:
        return list(set(source_links))    

    #otherwise return with probability 1/k*epoch_decay the k outgoing links
    pruned_links = []
    if len(links) != 0:
        link_prob = (1/max(len(links), max_outgoing_links)) * epoch_decay
    for link in links:
        if random.random() < link_prob:
            pruned_links.append(link)
    return list(set(pruned_links))

#Create the nth level of the diffusion graph
def graph_epoch_n(frontier, epoch, last_pass, twitter_corpus_file, diffusion_graph_dir):

    spark = init_spark()

    if epoch == 0:
        urlRegex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        documents = spark.textFile(twitter_corpus_file, minPartitions=(conf['partitions'])) \
        .map(lambda r: (lambda r: Row(source_url=r[0], tweet=r[1], timestamp=r[2], popularity=r[3], RTs=r[4], user_country=r[5]))(r.split('\t'))) \
        .flatMap(lambda r: [Row(source_url=r.source_url, timestamp=r.timestamp, popularity=r.popularity, RTs=r.RTs, user_country=r.user_country, target_url=resolve_short_url(u)) for u in re.findall(urlRegex, r.tweet) or ['']]) \
        .map(lambda r : '\t'.join(str(a) for a in [r.source_url, r.timestamp, r.popularity, r.RTs, r.user_country, r.target_url]))
        rdd2tsv(documents, diffusion_graph_dir+'epoch_'+str(epoch)+'.tsv', ['source_url','timestamp', 'popularity', 'RTs', 'user_country', 'target_url'])
    else:
        documents = spark.parallelize(frontier, numSlices=(conf['partitions'])) \
        .flatMap(lambda r: [Row(source_url=r, target_url=l) for l in get_out_links(r, epoch_decay=exp(-epoch), last_pass=last_pass) or ['']]) \
        .map(lambda r : '\t'.join(str(a) for a in [r.source_url, r.target_url]))
        rdd2tsv(documents, diffusion_graph_dir+'epoch_'+str(epoch)+'.tsv', ['source_url', 'target_url'])


#Create diffusion graph
def create_diffusion_graph(twitter_corpus_file, diffusion_graph_file):

    diffusion_graph_dir = '/'.join(diffusion_graph_file.split('/')[:-1])+'/'

    #initialize graph
    G=nx.DiGraph()

    for v in institutions['URL'].tolist():
        G.add_edge(v, graph_nodes['institution'])

    for v in repositories['URL'].tolist():
        G.add_edge(v, graph_nodes['repository'])

    G.add_edge(graph_nodes['institution'], graph_nodes['source'])
    G.add_edge(graph_nodes['repository'], graph_nodes['source'])

    epoch = 0
    frontier = []
    connected_components = 0
    last_pass = False
    while True:

        #expand graph
        if not os.path.exists(diffusion_graph_dir+'epoch_'+str(epoch)+'.tsv'):
            graph_epoch_n(frontier, epoch, last_pass, twitter_corpus_file, diffusion_graph_dir)

        df = pd.read_csv(diffusion_graph_dir+'epoch_'+str(epoch)+'.tsv', sep='\t').dropna()
        G =  nx.compose(G, nx.from_pandas_edgelist(df, source='source_url', target='target_url', create_using=nx.DiGraph()))
        frontier = [x for x in G.nodes() if G.out_degree(x) == 0]

        print('Epoch:', epoch)
        print('Connected Components:', nx.number_connected_components(G.to_undirected()))
        print('Frontier Size:', len(frontier))

        if last_pass:
            break
        
        #last pass condition
        if epoch != 0 and (connected_components - nx.number_connected_components(G.to_undirected())) / connected_components < components_ratio:
            last_pass = True
        connected_components = nx.number_connected_components(G.to_undirected())
        epoch +=1

    #add root node
    df = pd.read_csv(diffusion_graph_dir+'epoch_0.tsv', sep='\t').dropna()
    df['social'] = project_url+'#twitter'
    G =  nx.compose(G, nx.from_pandas_edgelist(df, source='social', target='source_url', create_using=nx.DiGraph()))

    write_graph(G, diffusion_graph_file)


#Recursively remove nodes that do not have an offspring
def clean_orphan_nodes(graph_in_file, graph_out_file):
    G = read_graph(graph_in_file)
    while True:
        nodes = [x for x in G.nodes() if (G.out_degree(x) == 0) and (x != project_url+'#source')]

        if len(nodes) == 0:
            break

        for n in nodes:
            G.remove_node(n)
    write_graph(G, graph_out_file)


#Download news/scientific article details
def download_documents(graph_in_file, graph_out_file, document_out_file, document_type, sleep_time=1):

    G = read_graph(graph_in_file)

    documents = []
    if document_type == 'paper':
        for r in G.predecessors(project_url+'#repository'):
            for p in G.predecessors(r):
                documents.append(p)

    elif document_type == 'article':
        for r in G.predecessors(project_url+'#repository'):
            for p in G.predecessors(r):
                for a in G.predecessors(p):
                    if 'https://twitter.com' not in a:
                        documents.append(a)

    documents = list(set(documents))

    document_details=[]
    removed_documents = []
    for i, d in enumerate(documents):
        print('\r %s%%' % ("{0:.2f}").format(100 * (i / float(len(documents)))), end = '\r')
        try:
            article = Article(d)
            article.download()
            article.parse()
            article.nlp()
            if len(article.text)>0:
                document_details.append([d, article.title, article.authors, article.keywords, article.publish_date, article.text])
            else:
                removed_documents.append(d)    
            sleep(sleep_time)
        except:
            removed_documents.append(d)
            continue

    document_details = pd.DataFrame(document_details, columns=['url','title','authors','keywords','publish_date','full_text'])
    document_details.to_csv(document_out_file, sep='\t', index=None)

    for n in removed_documents:
        G.remove_node(n)
    write_graph(G, graph_out_file)


#Get tweet details
def download_tweets(graph_file, twitter_corpus_file, twitter_out_file, sleep_time=1):

    G = read_graph(graph_file)

    tweets_details = pd.DataFrame([t for t in G.successors(project_url+'#twitter')]) 
    tweets_details = tweets_details.merge(pd.read_csv(twitter_corpus_file, sep='\t', header=None), left_on=0, right_on=0)
    tweets_details.columns = ['url','full_text','publish_date','popularity','RTs','user_country']
    
    tweets_details['replies'] = tweets_details['url'].apply(lambda x: scrap_twitter_replies(x, sleep_time))
    tweets_details['replies_num'] = tweets_details['replies'].apply(lambda x: len(x))
    tweets_details.to_csv(twitter_out_file, sep='\t', index=None)


#Remove articles that have the same text; tweets that point to that articles are redirected to one representative
def remove_duplicate_text(article_in_file, graph_in_file, article_out_file, graph_out_file):
    article_details = pd.read_csv(article_in_file, sep='\t')
    G = read_graph(graph_in_file)

    group = article_details.groupby('full_text')['url'].unique()
    duplicates = group[group.apply(lambda x: len(x)>1)].reset_index()['url'].tolist()

    remove_edges = []
    add_edges = []
    for d in duplicates:
        for l in d[1:]:
            for p in G.predecessors(l):
                G.add_edge(p, d[0])
            G.remove_node(l)
            article_details = article_details[article_details['url'] != l]

    write_graph(G, graph_out_file)
    article_details.to_csv(article_out_file, sep='\t', index=None)


#Filter graph based on the number of minimum domains referencing a publication (deprecated)
def filter_graph(graph_file, out_file, num_of_domains):
    G = read_graph(graph_file)

    pubs = []
    for r in G.predecessors(project_url+'#repository'):
        for n in G.predecessors(r):
            domains = set()
            for w in G.predecessors(n):
                domain, _ = analyze_url(w)
                domains.add(domain)
            pubs.append([n, len(domains)])
    pubs = pd.DataFrame(pubs)
    pubs = pubs.sort_values(1, ascending=False)

    pubs[pubs[1]>=num_of_domains][0].to_csv(out_file, index=False)


def create_corpus(corpus_file):
    diffusion_graph_dir = scilens_dir + 'cache/diffusion_graph/'+corpus_file.split('/')[-1].split('.')[-2]+'/'
    os.makedirs(diffusion_graph_dir, exist_ok=True)

    print('Step 1: Create Diffusion Graph')
    create_diffusion_graph(corpus_file, diffusion_graph_dir + 'diffusion_graph_v1.tsv')
    print('Step 2: Clean Orphan Nodes')
    clean_orphan_nodes(diffusion_graph_dir + 'diffusion_graph_v1.tsv', diffusion_graph_dir + 'diffusion_graph_v2.tsv')
    print('Step 3: Download Papers')
    download_documents(diffusion_graph_dir + 'diffusion_graph_v2.tsv', diffusion_graph_dir + 'diffusion_graph_v3.tsv', diffusion_graph_dir + 'paper_details_v1.tsv', 'paper')
    print('Step 4: Clean Orphan Nodes')
    clean_orphan_nodes(diffusion_graph_dir + 'diffusion_graph_v3.tsv', diffusion_graph_dir + 'diffusion_graph_v4.tsv')
    print('Step 5: Download News Articles')
    download_documents(diffusion_graph_dir + 'diffusion_graph_v4.tsv', diffusion_graph_dir + 'diffusion_graph_v5.tsv', diffusion_graph_dir + 'article_details_v1.tsv', 'article')
    print('Step 6: Clean Orphan Nodes')
    clean_orphan_nodes(diffusion_graph_dir + 'diffusion_graph_v5.tsv', diffusion_graph_dir + 'diffusion_graph_v6.tsv')
    print('Step 7: Download Tweets')
    download_tweets(diffusion_graph_dir + 'diffusion_graph_v6.tsv', corpus_file, diffusion_graph_dir + 'tweet_details_v1.tsv')
    print('Step 8: Remove Duplicate Articles')
    remove_duplicate_text(diffusion_graph_dir + 'article_details_v1.tsv', diffusion_graph_dir + 'diffusion_graph_v6.tsv', diffusion_graph_dir + 'article_details_v2.tsv', diffusion_graph_dir + 'diffusion_graph_v7.tsv')
