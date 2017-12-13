from spacy.symbols import nsubj, dobj, VERB
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.sql import Row

from settings import *
from utils import *

nlp, authorityKeywords, empiricalKeywords, actionsKeywords = initNLP()

##Pipeline functions
def quotePipeline():
    global spark
    spark = initSpark()
    documents = None

    t0 = time()
    if startPipelineFrom in ['start']:
        documents = cachefunc(extractQuotes, (documents))
    if startPipelineFrom in ['start', 'end']:
        topics = cachefunc(discoverTopics, (documents))
    
    print("Total time: %0.3fs." % (time() - t0))
    return documents, topics

def extractQuotes(documents):

    documents = readCorpus()

    #process articles to extract quotes
    documents = documents.map(lambda s: Row(article=s.article, quotes=dependencyGraphSearch(s.article)))
    
    #remove quotes from articles 
    documents = documents.map(lambda s: Row(article=removeQuotesFromArticle(s.article, s.quotes), quotes=s.quotes))

    #convert rdd to dataFrame
    documents = documents.toDF(['article', 'quotes'])
    #drop documents without quotes
    documents = documents.dropna()

    return documents

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
            
            #case where PERSON is referred to with his/her first or last name       
            if len(q.split()) == 1:
                for e in allPerEntities:
                    if q != e and q in e.split():
                        q = e
                        break
                        
            return (q, qtype, qaff)    

    #case that quotee ORG entity exists      
    for e in sOrgEntities:

        if e in quotee:
            q = e
            qtype = 'ORG'
            qaff = e
            
            #case where ORG is referred to with an acronym
            if len(q.split()) == 1:
                for e in allOrgEntities:
                    if q != e and len(e.split()) > 1:
                        fullAcronym = compactAcronym = upperAccronym = ''
                        for w in e.split():
                            for l in w:
                                if (l.isupper()):
                                    upperAccronym += l
                            if not nlp(w)[0].is_stop:
                                compactAcronym += w[0]
                            fullAcronym += w[0]

                        if q.lower() in [fullAcronym.lower(), compactAcronym.lower(), upperAccronym.lower()]:
                            q = e
                            qaff = e
                            break
       
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
    documents = documents.toPandas()
    
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
    print('Total number of quotes:',human_format(documents.shape[0]))
    print ('Average number of quotes per Document:',len(documents)/limitDocuments)

    #discover quote topics
    print('Discovering quote topics...')
    tf = tf_vectorizer.transform(documents['quote'])
    L = lda.transform(tf)

    documents['quoteTopic'] = [topicLabels[t] for t in L.argmax(axis=1)]
    documents['quoteSim'] = L.max(axis=1)

    topics = spark.createDataFrame(documents)
    return topics
