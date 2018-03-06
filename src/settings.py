#Use cached RDDs
useCache = True

#Memory used by Spark (8, 60 or 250)
memory = '8G'

#Corpus path 
corpusPath = '/home/psmeros/workspace/bigFiles/'
#corpusPath = '/Users/smeros/workspace/etc/bigFiles/'
#corpusPath = '/home/smeros/backup_data/'
#corpusPath = '/root/backup_data/'

webCorpusFile = corpusPath + 'webFood.tsv'
twitterCorpusFile = corpusPath + 'twitterFood.tsv'

#URL settings
urlTimout = 1
urlRegex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

#Topic Discovery parameters
numOfTopics = 32
topicTopfeatures = 1
max_iter = 20
samplingFraction = 0.2
topicSimThreshold = 0.5

#File with refined topics
topicsFile = 'auxiliary_files/topics/topics.txt'

#File with institutions metadata
institutionsFile = 'auxiliary_files/institutions/metadata.tsv'

#File with country codes
countriesFile = 'auxiliary_files/countries/codes.csv'

#File with academic repositories
repositoriesFile = 'auxiliary_files/repositories/academic_repositories.csv'

#Predefined keyword lists
authorityKeywords = ['expert', 'scientist', 'researcher', 'professor', 'author', 'paper', 'report', 'study', 'analysis', 'research', 'survey', 'release']
empiricalKeywords = ['study', 'people']
actionsKeywords = ['prove', 'demonstrate', 'reveal', 'state', 'mention', 'report', 'say', 'show', 'announce', 'claim', 'suggest', 'argue', 'predict', 'believe', 'think']

#Imports
import os
import re
import sys
import shutil
import builtins
from time import time
from datetime import datetime
import requests
from bs4 import BeautifulSoup
from urllib.request import urlopen
from urllib.parse import urlsplit
import numpy as np
import pandas as pd
import spacy
from spacy.symbols import nsubj, dobj, VERB
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics.pairwise import cosine_similarity
from pyspark.sql import SparkSession
from pyspark import SparkConf
from pyspark.sql.types import *
from pyspark.sql.functions import *
from pyspark.sql import Row

#Cache directory
os.makedirs('cache', exist_ok=True)

#Pandas settings
pd.set_option('display.max_colwidth', -1)
