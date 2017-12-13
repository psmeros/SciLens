import pandas as pd
import os

#Limit retrieved documents
limitDocuments = 100

memory = '8G'
cores = 4

#Pickled dataframe
useCache = False

#Starting point of the pipeline
startPipelineFrom ='start' #values: 'start', 'end'

#Topic Discovery parameters
numOfTopics = 32
topicTopfeatures = 1
max_iter = 5
samplingThreshold = 50000

#Corpus file
corpusFile = '/home/psmeros/workspace/bigFiles/sampleFoodArticles.tsv'
#corpusFile = '/Users/smeros/workspace/etc/bigFiles/sampleFoodArticles.tsv'
#corpusFile = '/home/smeros/backup_data/sampleFoodArticles.tsv'

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