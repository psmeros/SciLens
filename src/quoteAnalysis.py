from spacy.symbols import nsubj, dobj, VERB
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.sql import Row

from settings import *
from utils import *
from gloveEmbeddings import *


#Create Keyword Lists and SpaCy NLP object
nlp = English()
authorityKeywords = [nlp(x)[0].lemma_ for x in ['expert', 'scientist', 'researcher', 'professor', 'author', 'paper', 'report', 'study', 'analysis', 'research', 'survey', 'release']]
empiricalKeywords = [nlp(x)[0].lemma_ for x in ['study', 'people']]
actionsKeywords = [nlp(x)[0].lemma_ for x in ['prove', 'demonstrate', 'reveal', 'state', 'mention', 'report', 'say', 'show', 'announce', 'claim', 'suggest', 'argue', 'predict', 'believe', 'think']]


##Pipeline functions
def quotePipeline():
    global spark
    spark = initSpark()
    t0 = time()
    documents = None
    if startPipelineFrom in ['start']:
        documents = cachefunc(extractQuotes, (documents))
    # if startPipelineFrom in ['start', 'extractQuotes', 'removeQuotes', 'end']:
    #     documents = cachefunc(discoverTopics, (documents))
    print("Total time: %0.3fs." % (time() - t0))
    return documents

def extractQuotes(documents):

    documents = queryDB()

    #process articles to extract quotes
    #documents = documents.select('article', dependencyGraphSearchUDF('article').alias('quotes'))
    documents = documents.rdd.map(lambda s: Row(article=s.article, quotes=dependencyGraphSearch(s.article))).toDF(['article', 'quotes'])


    #drop documents without quotes
    documents = documents.dropna()
    
    #remove quotes from articles 
    #documents = documents.select(removeQuotesFromArticleUDF('article', 'quotes').alias('article'), 'quotes')
    documents = documents.rdd.map(lambda s: Row(article=removeQuotesFromArticle(s.article, s.quotes), quotes=s.quotes)).toDF(['article', 'quotes'])

    return documents

#UDF definitions
@udf(ArrayType(MapType(StringType(), StringType())))
def dependencyGraphSearchUDF(article):
    dependencyGraphSearch(article)


@udf(StringType())
def removeQuotesFromArticleUDF(article, quotes):
    removeQuotesFromArticle(article, quotes)


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

    documents = documents.toPandas()
    vocabulary = createVocabulary()
    
    #convert to tf vectors (1-2grams)
    tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english', ngram_range=(1,2), token_pattern='[a-zA-Z]{2,}', vocabulary=vocabulary['literal'].tolist())
    tf = tf_vectorizer.fit_transform(documents['article'])

    #fit lda topic model
    lda = LatentDirichletAllocation(n_components=numOfTopics, max_iter=20, learning_method='online', n_jobs=-1)
    lda.fit(tf)

    #get the names of the top features of each topic and map it to lower dimensions (concepts)
    labels = pd.DataFrame(tf_vectorizer.get_feature_names(), columns=['literal'])
    labels = labels.merge(vocabulary, left_on='literal', right_on='literal')['concept'].tolist()

    topiclabels = []
    for _, topic in enumerate(lda.components_):
        topiclabels.append([labels[i] for i in topic.argsort()[:-50 - 1:-1]])

    newlabels = []
    for t in topiclabels:
        t = pd.DataFrame(t, columns=['concept'])
        t = t.groupby('concept', sort=False).size().reset_index(name='size')#.sort_values(ascending=False)
        nl = ', '.join(t[t['size']>1].sort_values(by='size', ascending=False).head(topicTopfeatures)['concept'].tolist())
        
        rest = topicTopfeatures - len(t[t['size']>1])
        if rest == topicTopfeatures:
            nl += ', '.join(t.head(rest)['concept'].tolist())
        elif  rest > 0 :
            nl += ', ' + ', '.join(t.head(rest)['concept'].tolist())

        newlabels.append(nl)
    topiclabels = newlabels

    #add the topic label as a column in the dataFrame
    L = lda.transform(tf)
    documents['articleTopic'] = [topiclabels[t] for t in L.argmax(axis=1)]
    documents['articleSim'] = L.max(axis=1)

    print('Total number of topics:', len(documents['articleTopic'].unique()))

    #flatten quotes
    documents = documents[['articleTopic', 'articleSim']].join(documents['quotes'].apply(pd.Series).stack().reset_index(level=1, drop=True).apply(pd.Series))    
    print('Total number of quotes:',human_format(documents.shape[0]))
    print ('Average number of quotes per Document:',len(documents)/limitDocuments)

    #discover quote topics
    tf = tf_vectorizer.transform(documents['quote'])
    L = lda.transform(tf)

    documents['quoteTopic'] = [topiclabels[t] for t in L.argmax(axis=1)]
    documents['quoteSim'] = L.max(axis=1)

    documents = spark.createDataFrame(documents)
    return documents

