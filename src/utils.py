from settings import *

#Spark setup
def initSpark():
    global spark
    conf = SparkConf()
    conf.setAppName('quoteAnalysis')
    conf.setMaster('local['+str(cores)+']')
    conf.set('spark.executor.memory', str(int(memory/cores))+'G')
    conf.set('spark.driver.memory', str(int(memory/cores))+'G')
    conf.set('spark.hadoop.validateOutputSpecs', 'false')
    spark = SparkSession.builder.config(conf=conf).getOrCreate()
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
    for s in sorted(personKeywords + studyKeywords):
        for p in sorted(actionsKeywords):
            print(s, p)

#Write RDD to TSV file (with header)
def rdd2tsv(rdd, file, attributes):
    rdd.saveAsTextFile(file+'_files')
    os.system('echo "' + '\t'.join(attributes) + '" > ' + file + '; cat ' + file + '_files/* >> ' + file + '; rm -r ' + file + '_files')


#Find the domain and the path of an http url
def analyze_url(url):
    url=urlsplit(url)
    domain = re.sub(r'^(http(s)?://)?(www\.)?', r'', url.netloc)
    if domain.count('.') == 2:
        domain = ('.').join(domain.split('.')[1:])
    return domain, url.path

#diffusion graph filename
def get_graph_filename(epoch):
    return ('_epoch_'+str(epoch)+'.').join(diffusion_graph_file.split('.'))