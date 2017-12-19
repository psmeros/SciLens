from settings import *
from utils import *

nlp, authorityKeywords, empiricalKeywords, actionsKeywords = initNLP()

##Pipeline functions
def quotePipeline():
    global spark
    spark = initSpark()
    documents = None

    t00 = time()
    if startPipelineFrom in ['start']:
        func = extractQuotes
        cache_doc = 'cache/'+func.__name__+'_doc.pkl'
        cache_q = 'cache/'+func.__name__+'_q.pkl'
        if useCache and os.path.exists(cache_doc) and os.path.exists(cache_q):
            print ('Reading from cache:', cache_doc)
            documents = spark.sparkContext.pickleFile(cache_doc)
            print ('Reading from cache:', cache_q)
            quotes = spark.sparkContext.pickleFile(cache_q)
        else:
            t0 = time()
            documents, quotes = func()
            shutil.rmtree(cache_doc, ignore_errors=True)
            shutil.rmtree(cache_q, ignore_errors=True)
            documents.saveAsPickleFile(cache_doc)
            quotes.saveAsPickleFile(cache_q)
            print(func.__name__, "ran in %0.3fs." % (time() - t0))
    if startPipelineFrom in ['start', 'end']:
        func = discoverTopics
        cache = 'cache/'+func.__name__+'.pkl'
        if useCache and os.path.exists(cache):
            print ('Reading from cache:', cache)
            topics = spark.sparkContext.pickleFile(cache)
        else:
            t0 = time()
            topics = func(documents)
            shutil.rmtree(cache, ignore_errors=True)
            topics.saveAsPickleFile(cache)
            print(func.__name__, "ran in %0.3fs." % (time() - t0))
    
    print("Total time: %0.3fs." % (time() - t00))
    return documents, quotes, topics

def extractQuotes():

    documents = readCorpus()

    #process articles to extract quotes
    documents = documents.map(lambda s: Row(article=s.article, quotes=dependencyGraphSearch(s.article)))
    
    #remove quotes from articles 
    documents = documents.map(lambda s: Row(article=removeQuotesFromArticle(s.article, s.quotes), quotes=s.quotes))

    #drop documents without quotes
    documents = documents.filter(lambda s: s.article is not None)

    quotes = documents.flatMap(lambda s: [Row(quote=q['quote'], quotee=q['quotee'], quoteeType=q['quoteeType'], quoteeAffiliation=q['quoteeAffiliation']) for q in s.quotes])
    documents = documents.map(lambda s: Row(article=s.article, quotes=[q['quote']for q in s.quotes]))

    return documents, quotes

# Search for quote patterns
def dependencyGraphSearch(article):

    allPerEntities = []
    allOrgEntities = []
    for e in nlp(article).ents:
        if e.label_ == 'PERSON':
            allPerEntities.append(e.text)
        elif e.label_ == 'ORG':
            allOrgEntities.append(e.text)
            
    quotes = []

    for s in sent_tokenize(article):
        quoteFound = False
        quote = quotee = quoteeType = quoteeAffiliation = ""
        s = nlp(s)

        sPerEntities = []
        sOrgEntities = []
        for e in s.ents:
            if e.label_ == 'PERSON':
                sPerEntities.append(e.text)
            elif e.label_ == 'ORG':
                sOrgEntities.append(e.text)


        #find all verbs of the sentence.
        verbs = set()
        for v in s:
            if v.head.pos == VERB:
                verbs.add(v.head)

        if not verbs:
            continue

        rootVerb = ([w for w in s if w.head is w] or [None])[0]

        #check first the root verb and then the others.
        verbs = [rootVerb] + list(verbs)

        for v in verbs:
            if v.lemma_ in actionsKeywords:            

                quoteFound = True
                
                for np in v.children:
                    if np.dep == nsubj:
                        quotee = s[np.left_edge.i : np.right_edge.i+1].text
                        break

            if quoteFound:
                    quote = s.text.strip()
                    quotee, quoteeType, quoteeAffiliation = resolveQuotee(quotee, sPerEntities, sOrgEntities, allPerEntities, allOrgEntities)
                    quotes.append({'quote': quote, 'quotee':quotee, 'quoteeType':quoteeType, 'quoteeAffiliation':quoteeAffiliation})
                    break
    
    if quotes == []:
        return None
    else:
        return quotes

#Resolves the quotee of a quote.
def resolveQuotee(quotee, sPerEntities, sOrgEntities, allPerEntities, allOrgEntities):
    
    q = qtype = qaff = 'unknown'
    
    #case that quotee PER entity exists
    for e in sPerEntities:
        if e in quotee:
            q = e
            qtype = 'PERSON'
            
            #find affiliation of person
            for e in sOrgEntities:
                if e in quotee:
                    qaff = e
                    break
            
            p = resolvePerson(q, allPerEntities)
            if p != None:
                q = p
                        
            return (q, qtype, qaff)    

    #case that quotee ORG entity exists      
    for e in sOrgEntities:

        if e in quotee:
            q = e
            qtype = 'ORG'
            qaff = e
            
            o = resolveOrganization(q, allOrgEntities)
            if o != None:
                q = o
                qaff = o
       
            return (q, qtype, qaff)   
        
    #case that quotee entity doesn't exist
    try:
        noun = next(nlp(quotee).noun_chunks).root.lemma_
    except:    
        return (q, qtype, qaff)
    
    if noun in authorityKeywords:
        q = 'authority'
    elif noun in empiricalKeywords:
        q = 'empirical observation'
    return (q, qtype, qaff)

#Resolve cases where PERSON is referred to with his/her first or last name       
def resolvePerson(per, plist):
    if len(per.split()) == 1:
        for p in plist:
            if per != p and per in p.split():
                #print(per, ' to ', p)
                return p
    return None

#Resolve cases where ORG is referred to with an acronym
def resolveOrganization(org, olist):
    if len(org.split()) == 1:
        for o in olist:
            if org != o and len(o.split()) > 1:
                fullAcronym = compactAcronym = upperAccronym = ''
                for w in o.split():
                    for l in w:
                        if (l.isupper()):
                            upperAccronym += l
                    if not nlp(w)[0].is_stop:
                        compactAcronym += w[0]
                    fullAcronym += w[0]

                if org.lower() in [fullAcronym.lower(), compactAcronym.lower(), upperAccronym.lower()]:
                    #print(org, ' to ', o)
                    return o
    return None

# Remove quotes from articles
def removeQuotesFromArticle(article, quotes):
    if quotes == None:
        return None
    articleWithoutQuotes = ''
    it = iter(quotes)
    q = next(it)['quote']
    for s in sent_tokenize(article):
        s = s.strip()
        if (q and s == q):
            q = next(it, None)
            if q != None:
                q = q['quote']
        else:
            articleWithoutQuotes += s + ' '
    return articleWithoutQuotes


def discoverTopics(documents):

    #subsampling
    documents = documents.sample(withReplacement=False, fraction=samplingFraction)
    documents = documents.toDF(['article', 'quotes']).toPandas()
    
    vocabulary = createVocabulary()
    
    #define vectorizer (1-2grams)
    tf_vectorizer = CountVectorizer(max_df=0.8, min_df=0.2, stop_words='english', ngram_range=(1,2), token_pattern='[a-zA-Z]{2,}', vocabulary=vocabulary)
    tf = tf_vectorizer.transform(documents['article'])

    #fit lda topic model
    print('Fitting LDA model...')
    lda = LatentDirichletAllocation(n_components=numOfTopics, max_iter=max_iter, learning_method='online', n_jobs=cores)
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
    print('Sample quotes:',human_format(documents.shape[0]))
    print ('Average number of quotes per Document:',len(documents)/limitDocuments)

    #discover quote topics
    print('Discovering quote topics...')
    tf = tf_vectorizer.transform(documents['quote'])
    L = lda.transform(tf)

    documents['quoteTopic'] = [topicLabels[t] for t in L.argmax(axis=1)]
    documents['quoteSim'] = L.max(axis=1)

    topics = spark.createDataFrame(documents).rdd
    return topics
