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
    while abs(num) >= 1000:
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
