from settings import *

#Spark setup
def initSpark():
    global spark
    conf = SparkConf()
    conf.setAppName('quoteAnalysis')
    conf.setMaster('local[*]')
    conf.set('spark.executor.memory', memory)
    conf.set('spark.driver.memory', memory)
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
    for s in authorityKeywords + empiricalKeywords:
        for p in actionsKeywords:
            print('"'+s, p+'"')



            