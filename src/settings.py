#Memory in GBs - cores used by Spark and cache folder
#conf = {'memory':8, 'cores':4, 'partitions':4*20, 'aux_dir':'/home/psmeros/Dropbox/scilens/'} #batfink
conf = {'memory':8, 'cores':4, 'partitions':4*20, 'aux_dir':'/Users/smeros/Dropbox/scilens/'} #mac
#conf = {'memory':64, 'cores':24, 'partitions':24*20, 'aux_dir': '/home/smeros/backup_data/scilens/'}    #lsir-cloud
#conf = {'memory':252, 'cores':48, 'partitions':48*20, 'corpusPath': '/root/'}  #iccluster

#Use cached files
useCache = True
cache_dir = conf['aux_dir'] + 'cache/'

#Corpus files
twitterCorpusFile = conf['aux_dir'] + 'corpus/scilens_3M.tsv'
glove_file = conf['aux_dir'] + 'big_files/glove.6B.300d.txt'
twitter_users_file = conf['aux_dir'] + 'corpus/twitter_users.tsv'

#Graph settings
project_url = 'http://sci-lens.org'
graph_nodes = {'tweetWithoutURL':project_url+'#tweetWithoutURL', 'HTTPError':project_url+'#HTTPError', 'TimeoutError':project_url+'#TimeoutError', 'institution':project_url+'#institution', 'repository':project_url+'#repository', 'source':project_url+'#source'}


#URL redirection settings
url_timeout = 1

#Components ratio for graph construction
components_ratio = 0.10

#Topic Discovery parameters
numOfTopics = 16
max_iter = 100

#Auxiliary Files
#File with refined topics
hn_vocabulary = open(conf['aux_dir'] + 'small_files/hn_vocabulary/hn_vocabulary.txt').read().splitlines()
#File with institutions metadata
institutionsFile = conf['aux_dir'] + 'small_files/institutions/metadata.tsv'
#File with country codes
countriesFile = conf['aux_dir'] + 'small_files/countries/codes.csv'
#File with academic repositories
repositoriesFile = conf['aux_dir'] + 'small_files/repositories/academic_repositories.csv'
#blacklisted URLs
blacklistURLs = open(conf['aux_dir'] + 'small_files/blacklist/urls.txt').read().splitlines()
#Predefined keyword lists
personKeywordsFile = conf['aux_dir'] + 'small_files/keywords/person.txt'
studyKeywordsFile = conf['aux_dir'] + 'small_files/keywords/study.txt'
actionsKeywordsFile = conf['aux_dir'] + 'small_files/keywords/action.txt'