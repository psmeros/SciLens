#Memory in GBs - cores used by Spark and corpus path
conf = {'memory':8, 'cores':4, 'corpusPath':'/Users/smeros/workspace/etc/bigFiles/'}
#conf = {'memory':8, 'cores':4, 'corpusPath':'/home/psmeros/workspace/bigFiles/'}
#conf = {'memory':64, 'cores':24, 'corpusPath': '/home/smeros/backup_data/'}
#conf = {'memory':252, 'cores':48, 'corpusPath': '/root/'}

#Use cached files
useCache = True

#Corpus files
webCorpusFile = conf['corpusPath'] + 'webFood.tsv'
twitterCorpusFile = conf['corpusPath'] + 'twitterFoodSample.tsv'

#Graph files
diffusion_graph_dir = 'cache/diffusion_graph/'

#URL redirection settings
urlTimout = 1

#Components ratio for graph construction
components_ratio = 0.1

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
#blacklisted URLs
blacklistURLsFile = 'auxiliary_files/blacklist/urls.txt'
#Predefined keyword lists
personKeywordsFile = 'auxiliary_files/keywords/person.txt'
studyKeywordsFile = 'auxiliary_files/keywords/study.txt'
actionsKeywordsFile = 'auxiliary_files/keywords/action.txt'

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
#from networkx.algorithms import bipartite

#Cache directories
os.makedirs('cache', exist_ok=True)
os.makedirs(diffusion_graph_dir, exist_ok=True)

#Pandas settings
pd.set_option('display.max_colwidth', -1)
