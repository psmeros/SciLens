#Scilens Directory
from pathlib import Path
scilens_dir = str(Path.home()) + '/Dropbox/scilens/'

#Use cached files
useCache = True
cache_dir = scilens_dir + 'cache/'

#Corpus files
twitterCorpusFile = scilens_dir + 'corpus/scilens_3M.tsv'
glove_file = scilens_dir + 'big_files/glove.6B.300d.txt'
twitter_users_file = scilens_dir + 'corpus/twitter_users.tsv'


#Topic Discovery parameters
numOfTopics = 16
max_iter = 100

#Minimum length for articles/paragraphs/sentences (#chars)
MIN_ART_LENGTH = 256
MIN_PAR_LENGTH = 256
MIN_SEN_LENGTH = 32

#Auxiliary Files
#File with refined topics
hn_vocabulary = open(scilens_dir + 'small_files/hn_vocabulary/hn_vocabulary.txt').read().splitlines()
#File with country codes
countriesFile = scilens_dir + 'small_files/countries/codes.csv'
#Predefined keyword lists
personKeywordsFile = scilens_dir + 'small_files/keywords/person.txt'
studyKeywordsFile = scilens_dir + 'small_files/keywords/study.txt'
actionsKeywordsFile = scilens_dir + 'small_files/keywords/action.txt'