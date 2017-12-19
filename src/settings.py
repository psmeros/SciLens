import os
import re
import sys
import shutil
import builtins
from time import time
import numpy as np
import pandas as pd
from spacy.en import English
from spacy.symbols import nsubj, dobj, VERB
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from bs4 import BeautifulSoup
from pyspark.sql import SparkSession
from pyspark import SparkConf
from pyspark.sql.types import *
from pyspark.sql.functions import *
from pyspark.sql import Row

#Limit retrieved documents
limitDocuments = 100

memory = '8G' #8 or 60
cores = 4 #4 or 24

#Pickled dataframe
useCache = False

#Starting point of the pipeline
startPipelineFrom ='start' #'start' or 'end'

#Topic Discovery parameters
numOfTopics = 32
topicTopfeatures = 1
max_iter = 5
samplingFraction = 0.9
topicSimThreshold = 0.7

#Corpus file
#corpusFile = None
corpusFile = '/home/psmeros/workspace/bigFiles/sampleFoodArticles.tsv'
#corpusFile = '/Users/smeros/workspace/etc/bigFiles/sampleFoodArticles.tsv'
#corpusFile = '/home/smeros/backup_data/FoodArticles.tsv'

#GloVe Embeddings file
gloveFile = None
#gloveFile = '/Users/smeros/workspace/etc/bigFiles/glove.6B/glove.6B.300d.txt'
#gloveFile = '/home/psmeros/workspace/bigFiles/glove.6B.300d.txt'
#gloveFile = '/home/smeros/glove_data/glove.6B.300d.txt'

#Cache and plots directory
os.makedirs('cache', exist_ok=True)
os.makedirs('plots', exist_ok=True)

#File with refined topics
topicsFile = 'topics.txt'

#Pandas settings
pd.set_option('display.max_colwidth', -1)