from pyspark.sql import SparkSession
from pyspark import SparkConf

from settings import *

#Spark setup
def initSpark():
    global spark
    spark_conf = SparkConf()
    spark_conf.setAppName('quoteAnalysis')
    spark_conf.setMaster('local['+str(conf['cores'])+']')
    spark_conf.set('spark.executor.memory', str(int(conf['memory']/conf['cores']))+'G')
    spark_conf.set('spark.driver.memory', str(int(conf['memory']/conf['cores']))+'G')
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
    url=urlsplit(url)
    domain = re.sub(r'^(http(s)?://)?(www\.)?', r'', url.netloc)
    path = '' if domain == '' else url.path

    return domain, path

#Compare two domains
def same_domains(domain_1, domain_2):
    if domain_1.count('.') == 2:
        domain_1 = ('.').join(domain_1.split('.')[1:])
    if domain_2.count('.') == 2:
        domain_2 = ('.').join(domain_2.split('.')[1:])
    
    if domain_1 in domain_2 or domain_2 in domain_1:
        return True
    return False
