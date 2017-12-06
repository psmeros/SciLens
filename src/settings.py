import pandas as pd
import os

#Limit retrieved documents
limitDocuments = 10

memory = '8G'
cores = 4

#Pickled dataframe
useCache = True

#Starting point of the pipeline
startPipelineFrom ='start' #values: 'start', 'end'

#Topic Discovery parameters
numOfTopics = 32
topicTopfeatures = 3

#Settings for database connection
dbSettings = {'user':'smeros', 'password':'vasoula', 'db':'sciArticles', 'host':'localhost', 'port':'5432'}

#GloVe Embeddings file
gloveFile = None
#gloveFile = '/Users/smeros/workspace/etc/bigFiles/glove.6B/glove.6B.300d.txt'
#gloveFile = '/home/psmeros/workspace/bigFiles/glove.6B.300d.txt'
#gloveFile = '/home/smeros/glove_data/glove.6B.300d.txt'

#Cache and plots directory
os.makedirs('cache', exist_ok=True)
os.makedirs('plots', exist_ok=True)

#File with refined concepts
conceptsFile = 'concepts.csv'

#Pandas settings
pd.set_option('display.max_colwidth', -1)
