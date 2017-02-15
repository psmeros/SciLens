import numpy as np
import pandas as pd

import os
import pickle
import csv
import itertools

from sklearn import svm, linear_model, preprocessing, decomposition
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.neural_network import MLPClassifier 
from sklearn.naive_bayes import MultinomialNB

import sqlalchemy


def connect(user, password, db, host='localhost', port=5432):
    '''Returns a connection and a metadata object'''
    # We connect with the help of the PostgreSQL URL
    # postgresql://federer:grandestslam@localhost:5432/tennis
    url = 'postgresql://{}:{}@{}:{}/{}'
    url = url.format(user, password, host, port, db)

    # The return value of create_engine() is our connection object
    con = sqlalchemy.create_engine(url, client_encoding='utf8')

    # We then bind the connection to MetaData()
    meta = sqlalchemy.MetaData(bind=con, reflect=True)

    return con, meta



# Load Data
print('Loading data')

con, meta = connect('smeros', '', 'sciArticles')

sql = 'select body from document'
tweets = pd.read_sql(sql, con, index_col=None, coerce_float=True, params=None, parse_dates=None, columns=None, chunksize=None)

#tweets = pd.DataFrame(read_file(POS_TWEETS_FILE), columns=['tweet'])
#tweets['sentiment'] = 1

# Data Shape
#print('\ttweets shape: ',tweets.shape)


