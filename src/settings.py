#Memory in GBs - cores used by Spark and corpus path
#conf = {'memory':8, 'cores':4, 'corpusPath':'/Users/smeros/workspace/etc/bigFiles/'}
conf = {'memory':8, 'cores':4, 'corpusPath':'/home/psmeros/workspace/bigFiles/'}
#conf = {'memory':64, 'cores':24, 'corpusPath': '/home/smeros/backup_data/'}
#conf = {'memory':252, 'cores':48, 'corpusPath': '/root/'}

memory = conf['memory']
cores = conf['cores']
corpusPath = conf['corpusPath']

#Use cached files
useCache = True

#Corpus files
webCorpusFile = corpusPath + 'webFood.tsv'
twitterCorpusFile = corpusPath + 'twitterFoodSampleSample.tsv'

#Graph files
first_level_graph_file = 'cache/first_level_graph.tsv'
second_level_urls_file = 'cache/second_level_urls.tsv'
second_level_graph_file = 'cache/second_level_graph.tsv'

#URL settings
urlTimout = 1
urlRegex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

#Topic Discovery parameters
numOfTopics = 32
topicTopfeatures = 1
max_iter = 20
samplingFraction = 0.2
topicSimThreshold = 0.5

#Auxiliary Files
#File with refined topics
topicsFile = 'auxiliary_files/topics/topics.txt'
#File with institutions metadata
institutionsFile = 'auxiliary_files/institutions/metadata.tsv'
#File with country codes
countriesFile = 'auxiliary_files/countries/codes.csv'
#File with academic repositories
repositoriesFile = 'auxiliary_files/repositories/academic_repositories.csv'
#blacklist URLs
blacklistFile = 'auxiliary_files/blacklist/urls.txt'

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
import networkx as nx
from networkx.algorithms import bipartite

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
