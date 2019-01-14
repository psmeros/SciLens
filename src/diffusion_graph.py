import os
import random
import re
from datetime import datetime
from math import ceil, exp, floor
from urllib.parse import urlsplit
from urllib.request import urlopen

import networkx as nx
import pandas as pd
import requests
from bs4 import BeautifulSoup
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

        return r.url

    #Catch the different errors       
    except requests.HTTPError as e:
        return graph_nodes['HTTPError']
    except:
        return graph_nodes['TimeoutError']

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
            links.append(link)
            for s in sources:
                if s in link_domain:
                    source_links.append(link)

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
def graph_epoch_n(frontier, epoch, last_pass, twitter_corpus_file):

    spark = init_spark()

    if epoch == 0:
        urlRegex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        documents = spark.textFile(twitter_corpus_file, minPartitions=(conf['partitions'])) \
        .map(lambda r: (lambda r: Row(source_url=r[0], tweet=r[1], timestamp=datetime.strptime(r[2], '%Y-%m-%d %H:%M:%S'), popularity=int(r[3]), RTs=int(r[4]), user_country=r[5]))(r.split('\t'))) \
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
            graph_epoch_n(frontier, epoch, last_pass, twitter_corpus_file)

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
