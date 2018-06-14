from pyspark.sql import SparkSession
from pyspark import SparkConf
from math import floor, ceil

from settings import *

#Spark setup
def initSpark():
    global spark
    spark_conf = SparkConf()
    spark_conf.setAppName('diffusion graph')
    spark_conf.setMaster('local['+str(conf['partitions'])+']')
    spark_conf.set('spark.executor.memory', str(floor(conf['memory']*0.9))+'G')
    spark_conf.set('spark.driver.memory', str(ceil(conf['memory']*0.1))+'G')
    spark_conf.set('spark.hadoop.validateOutputSpecs', 'false')
    spark = SparkSession.builder.config(conf=spark_conf).getOrCreate()
    spark.sparkContext.setLogLevel('ERROR')
    return spark

#Pretty print of numbers (by https://stackoverflow.com/a/45846841)
def human_format(num):
    num = float('{:.3g}'.format(num))
    magnitude = 0
    while builtins.abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    return '{}{}'.format('{:f}'.format(num).rstrip('0').rstrip('.'), ['', 'K', 'M', 'B', 'T'][magnitude])

#SEMPI keywords
def create_crawl_keywords():
    personKeywords = open(personKeywordsFile).read().splitlines()
    studyKeywords = open(studyKeywordsFile).read().splitlines()
    actionsKeywords = open(actionsKeywordsFile).read().splitlines()
    for s in sorted(personKeywords + studyKeywords):
        for p in sorted(actionsKeywords):
            print(s, p)

#Write RDD to TSV file (with header)
def rdd2tsv(rdd, file, attributes):
    rdd.saveAsTextFile(file+'_files')
    os.system('echo "' + '\t'.join(attributes) + '" > ' + file + '; cat ' + file + '_files/* >> ' + file + '; rm -r ' + file + '_files')


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

#Plot helpers (not working)
def plot_helper():
    df = pd.read_csv(diffusion_graph_dir+'epoch_0.tsv', sep='\t').dropna()    
    tweets = df.copy().drop('target_url', axis=1).drop_duplicates('source_url')
    #beutify country names
    tweets = tweets.merge(pd.read_csv(countriesFile).rename(columns={'Name':'Country'}), left_on='user_country', right_on='Code').drop(['user_country', 'Code'], axis=1).set_index('source_url')
    tweets.loc[tweets['Country'] == 'United States', 'Country'] = 'USA'
    print('Initial Tweets:', len(tweets))

    #Popularity
    inst.groupby('Institution').mean()['popularity'].sort_values(ascending=False)[:20]
    repos.groupby('Field').size().sort_values(ascending=False)
    inst.groupby('Institution').mean().plot.scatter(x='Score', y='popularity')
    corr = inst.groupby('Institution').mean()[['popularity', 'World Rank', 'National Rank', 'Alumni Employment', 'Publications', 'Influence', 'Citations', 'Broad Impact', 'Patents', 'Score']].corr()
    #sns.heatmap(corr, xticklabels=corr.columns.values, yticklabels=corr.columns.values)
    corr.iloc[0]

    #bipartite graph
    countries['Name'] = countries['Name'].map(lambda n: n+'_user')
    countries['Location'] = countries['Location'].map(lambda n: n+'_inst')
    B = nx.Graph()
    B.add_edges_from([(row['Name'], row['Location']) for _, row in countries.iterrows()])
    plt.figure(figsize=(10,10))
    X, Y = bipartite.sets(B)
    pos = dict()
    pos.update( (n, (1, i)) for i, n in enumerate(X) ) # put nodes from X at x=1
    pos.update( (n, (2, i*4)) for i, n in enumerate(Y) ) # put nodes from Y at x=2
    nx.draw(B, pos=pos, with_labels = True)

#Plot URL decay per year
def plot_URL_decay():
    df = pd.read_csv(diffusion_graph_dir+'epoch_0.tsv', sep='\t')
    df['date'] = df['timestamp'].apply(lambda s : datetime.strptime(s, '%Y-%m-%d %H:%M:%S').year)
    df['target_url'] = df['target_url'].apply(lambda u: u if u in [graph_nodes['tweetWithoutURL'], graph_nodes['HTTPError'], graph_nodes['TimeoutError']] else 'working URL')
    df['Tweets with'] = df['target_url'].map(lambda n: 'HTTP error in outgoing URL' if n == graph_nodes['HTTPError'] else 'timeout error in outgoing URL' if n == graph_nodes['TimeoutError'] else 'no URL' if n == graph_nodes['tweetWithoutURL'] else 'working URL')
    df[['source_url', 'date','Tweets with']].pivot_table(index='date', columns='Tweets with',aggfunc='count').T.reset_index(level=0, drop=True).T.fillna(1).plot(logy=True, figsize=(10,10), sort_columns=True)
