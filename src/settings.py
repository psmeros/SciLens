import pandas as pd
import logging
import sys
import os


#Pickled dataframe
useCache = False

#Use Spark for parallel processing
useSpark = True

#Limit retrieved documents
limitDocuments = 10

#Settings for database connection
dbSettings = {'user':'smeros', 'password':'vasoula', 'db':'sciArticles', 'host':'localhost', 'port':5432}

#GloVe Embeddings file
gloveFile = '/Users/smeros/workspace/etc/bigFiles/glove.6B/glove.6B.300d.txt'
#gloveFile = '/home/psmeros/var/workspaces/nutrition-workspace/bigFiles/glove.6B.300d.txt'
#gloveFile = '/home/smeros/glove_data/glove.6B.300d.txt'

if not os.path.exists(gloveFile):
    print(gloveFile,'embeddings not found')
    sys.exit(0)

#Cache directory
os.makedirs('cache', exist_ok=True)
 
#Plots directory
os.makedirs('plots', exist_ok=True)

#Pandas Settings
pd.set_option('display.max_colwidth', 1024)
