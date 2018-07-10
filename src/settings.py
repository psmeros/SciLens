#Memory in GBs - cores used by Spark and corpus path
#conf = {'memory':8, 'cores':4, 'partitions':4*20, 'corpusPath':'/Users/smeros/workspace/corpora/'}
conf = {'memory':8, 'cores':4, 'partitions':4*20, 'corpusPath':'/home/psmeros/workspace/corpora/'}
#conf = {'memory':64, 'cores':24, 'partitions':24*20, 'corpusPath': '/home/smeros/backup_data/'}
#conf = {'memory':252, 'cores':48, 'partitions':48*20, 'corpusPath': '/root/'}

#Use cached files
useCache = True

#Corpus files
#webCorpusFile = conf['corpusPath'] + 'webFood.tsv'
twitterCorpusFile = conf['corpusPath'] + 'scilens_3M.tsv'

#Graph files
diffusion_graph_dir = 'cache/diffusion_graph/'
project_url = 'http://sci-lens.org'
graph_nodes = {'tweetWithoutURL':project_url+'#tweetWithoutURL', 'HTTPError':project_url+'#HTTPError', 'TimeoutError':project_url+'#TimeoutError', 'institution':project_url+'#institution', 'repository':project_url+'#repository', 'source':project_url+'#source'}


#URL redirection settings
url_timeout = 1

#Components ratio for graph construction
components_ratio = 0.10

#selected papers
selected_papers_file = 'cache/selected_papers.txt'

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
blacklistURLs = open('auxiliary_files/blacklist/urls.txt').read().splitlines()
#Predefined keyword lists
personKeywordsFile = 'auxiliary_files/keywords/person.txt'
studyKeywordsFile = 'auxiliary_files/keywords/study.txt'
actionsKeywordsFile = 'auxiliary_files/keywords/action.txt'

#Cache directories
import os
os.makedirs('cache', exist_ok=True)
os.makedirs(diffusion_graph_dir, exist_ok=True)

