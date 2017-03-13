import pandas as pd

#SETTINGS
#Removes duplicate links from a document
removeDuplicateLinks = True

# Pandas Settings
pd.set_option('display.width', 1024)

# Pose query to db DB
def queryDB(query, user, password, db, host='localhost', port=5432):
    import sqlalchemy
    import warnings

    with warnings.catch_warnings():
        '''Returns a connection and a metadata object'''
        
        #ignore warning
        warnings.simplefilter("ignore", category=sqlalchemy.exc.SAWarning)
        
        # We connect with the help of the PostgreSQL URL
        url = 'postgresql://{}:{}@{}:{}/{}'
        url = url.format(user, password, host, port, db)

        # The return value of create_engine() is our connection object
        con = sqlalchemy.create_engine(url, client_encoding='utf8')

        # We then bind the connection to MetaData()
        #meta = sqlalchemy.MetaData(bind=con, reflect=True)

        #results as pandas dataframe
        results = pd.read_sql_query(query, con, index_col=None, coerce_float=True, params=None, parse_dates=None, chunksize=None)
        
        return results
