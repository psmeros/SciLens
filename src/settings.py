import pandas as pd

# Global Settings
pd.set_option('display.width', 1024)

# Connect to DB
def connectToDB(user, password, db, host='localhost', port=5432):
    import sqlalchemy
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=sqlalchemy.exc.SAWarning)
        '''Returns a connection and a metadata object'''
        # We connect with the help of the PostgreSQL URL
        url = 'postgresql://{}:{}@{}:{}/{}'
        url = url.format(user, password, host, port, db)

        # The return value of create_engine() is our connection object
        con = sqlalchemy.create_engine(url, client_encoding='utf8')

        # We then bind the connection to MetaData()
        meta = sqlalchemy.MetaData(bind=con, reflect=True)

    return con, meta

#Removes duplicate links from a document
removeDuplicateLinks = True