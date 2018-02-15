#Limit retrieved documents
limitDocuments = -1

memory = '8G' #8 or 60 or 250

#Pickled dataframe
useCache = True

#Starting point of the pipeline
runFromPipeline ='all' # 'all', 'extract'

#Topic Discovery parameters
numOfTopics = 32
topicTopfeatures = 1
max_iter = 20
samplingFraction = 0.2
topicSimThreshold = 0.5

#Corpus path 
corpusPath = '/home/psmeros/workspace/bigFiles/'
#corpusPath = '/Users/smeros/workspace/etc/bigFiles/'
#corpusPath = '/home/smeros/backup_data/'
#corpusFile = '/root/backup_data/'


corpusFile = corpusPath + 'foodArticles.tsv'
twitterCorpusFile = corpusPath + 'twitterFood.tsv'
urlTimout = 1


urlRegex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

#File with refined topics
topicsFile = 'auxiliary_files/topics/topics.txt'


import os
import re
import sys
import shutil
import builtins
from time import time
from datetime import datetime
import requests
import numpy as np
import pandas as pd
import spacy
from spacy.symbols import nsubj, dobj, VERB
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics.pairwise import cosine_similarity
from bs4 import BeautifulSoup
from pyspark.sql import SparkSession
from pyspark import SparkConf
from pyspark.sql.types import *
from pyspark.sql.functions import *
from pyspark.sql import Row

#Cache directory
os.makedirs('cache', exist_ok=True)

#Pandas settings
pd.set_option('display.max_colwidth', -1)
