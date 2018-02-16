from settings import *
from utils import initSpark

#Discover topics for both articles and quotes
def discoverTopics(documents):

    spark = initSpark()

    cache = 'cache/'+sys._getframe().f_code.co_name+'.pkl'

    if useCache and os.path.exists(cache):
        print ('Reading from cache:', cache)
        topics = spark.sparkContext.pickleFile(cache)
    else:
        t0 = time()

        #subsampling
        documents = documents.sample(withReplacement=False, fraction=samplingFraction)
        documents = documents.toDF(['article', 'quotes']).toPandas()
        
        #use topics as vocabulary to reduce dimensions
        if not os.path.exists(topicsFile):
            print(topicsFile,'topics not found!')
            sys.exit(0)

        with open(topicsFile) as f:
            vocabulary = f.read().splitlines()
        
        #define vectorizer (1-2grams)
        tf_vectorizer = CountVectorizer(max_df=0.8, min_df=0.2, stop_words='english', ngram_range=(1,2), token_pattern='[a-zA-Z]{2,}', vocabulary=vocabulary)
        tf = tf_vectorizer.transform(documents['article'])

        #fit lda topic model
        print('Fitting LDA model...')
        lda = LatentDirichletAllocation(n_components=numOfTopics, max_iter=max_iter, learning_method='online', n_jobs=-1)
        lda.fit(tf)

        #get the topic labels
        feature_names = tf_vectorizer.get_feature_names()
        topicLabels = []
        for _, topic in enumerate(lda.components_):
            topicLabels.append(" ".join([feature_names[i] for i in topic.argsort()[:-topicTopfeatures - 1:-1]]))

        #add the topic label as a column in the dataFrame
        print('Discovering article topics...')
        L = lda.transform(tf)
        documents['articleTopic'] = [topicLabels[t] for t in L.argmax(axis=1)]
        documents['articleSim'] = L.max(axis=1)

        #flatten quotes
        documents = documents[['articleTopic', 'articleSim']].join(documents['quotes'].apply(pd.Series).stack().reset_index(level=1, drop=True).apply(pd.Series))    
        documents.columns = ['articleTopic', 'articleSim', 'quote']

        #discover quote topics
        print('Discovering quote topics...')
        tf = tf_vectorizer.transform(documents['quote'])
        L = lda.transform(tf)

        documents['quoteTopic'] = [topicLabels[t] for t in L.argmax(axis=1)]
        documents['quoteSim'] = L.max(axis=1)

        topics = spark.createDataFrame(documents).rdd

        shutil.rmtree(cache, ignore_errors=True)
        topics.saveAsPickleFile(cache)
        print(sys._getframe().f_code.co_name, 'ran in %0.3fs.' % (time() - t0))
    
    return topics
