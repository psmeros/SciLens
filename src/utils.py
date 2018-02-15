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

#Create Keyword Lists and SpaCy NLP object
def initNLP():
    nlp = spacy.load('en')
    authorityKeywords = [nlp(x)[0].lemma_ for x in ['expert', 'scientist', 'researcher', 'professor', 'author', 'paper', 'report', 'study', 'analysis', 'research', 'survey', 'release']]
    empiricalKeywords = [nlp(x)[0].lemma_ for x in ['study', 'people']]
    actionsKeywords = [nlp(x)[0].lemma_ for x in ['prove', 'demonstrate', 'reveal', 'state', 'mention', 'report', 'say', 'show', 'announce', 'claim', 'suggest', 'argue', 'predict', 'believe', 'think']]
    return nlp, authorityKeywords, empiricalKeywords, actionsKeywords


#Read the corpus to the memory
def readCorpus():
    documents = spark.read.option('sep', '\t').csv(corpusFile, header=False, schema=StructType([StructField('article', StringType()),StructField('publishing_date', StringType()),StructField('url', StringType())]))
    documents = documents.limit(limitDocuments) if(limitDocuments!=-1) else documents

    return documents.rdd


#Pretty print of numbers (by https://stackoverflow.com/a/45846841)
def human_format(num):
    num = float('{:.3g}'.format(num))
    magnitude = 0
    while builtins.abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    return '{}{}'.format('{:f}'.format(num).rstrip('0').rstrip('.'), ['', 'K', 'M', 'B', 'T'][magnitude])