from spacy.en import English
from spacy.symbols import nsubj, dobj, VERB

from settings import *
from utils import *
from gloveEmbeddings import *

from pyspark.sql import SQLContext
from pyspark import SparkConf, SparkContext
if useSpark: ctx = SQLContext(SparkContext(conf = (SparkConf().setMaster('local[*]').setAppName('quoteExtraction').set('spark.executor.memory', '2G').set('spark.driver.memory', '40G').set('spark.driver.maxResultSize', '10G'))))

##Pipeline functions
def quotePipeline():
    documents = cachefunc(queryDB, ('web'))
    documents = cachefunc(extractQuotes, (documents))
    documents = cachefunc(flattenQuotes, (documents))    
    return documents

def extractQuotes(documents):
    #concatenation of title and body
    documents['article'] = documents['title'] + '.\n ' + documents['body']
    documents = documents.drop(['title', 'body'], axis=1)

    #process articles to extract quotes
    if useSpark:
        rddd = ctx.createDataFrame(documents[['article']]).rdd
        documents['quotes'] = rddd.map(lambda s: dependencyGraphSearch(s.article)).collect()
    else:
        documents['quotes'] = documents['article'].map(dependencyGraphSearch)

    print('Dropping '+ str(np.count_nonzero(documents['quotes'].isnull())) + ' document(s) without quotes.')
    documents = documents.dropna()
    
    return documents

def flattenQuotes(documents):
    documents = documents[['article', 'topic_label']].join(documents['quotes'].apply(pd.Series).stack().reset_index(level=1, drop=True).apply(pd.Series))    
    print('Total number of quotes:',human_format(documents.shape[0]))
    print ('Average number of quotes per Document:',len(documents)/limitDocuments)
    return documents

def discoverTopics(documents):
    #print('Total number of topics:', len(documents.topic_label.unique()))
    #discover the topic of each quote
    #topics = documents.topic_label.unique()
    #tEmbeddings = topics2Vec(topics)
    #documents['quoteTopic'], documents['sim'] = zip(*documents['quote'].map(lambda x: findQuoteTopic(x, tEmbeddings)))
    return documents

# Search for quote patterns
def dependencyGraphSearch(article):
    
    # Create Keyword Lists
    global nlp, sourcesKeywords, peopleKeywords, actionsKeywords
    try: 
        nlp('')
    except:
        nlp = English()
        sourcesKeywords = [nlp(x)[0].lemma_ for x in ['paper', 'report', 'study', 'analysis', 'research', 'survey', 'release']]
        peopleKeywords = [nlp(x)[0].lemma_ for x in ['expert', 'scientist']]
        actionsKeywords = [nlp(x)[0].lemma_ for x in ['prove', 'demonstrate', 'reveal', 'state', 'mention', 'report', 'say', 'show', 'announce', 'claim', 'suggest', 'argue', 'predict', 'believe', 'think']]
    
    allEntities = nlp(article).ents
    quotes = []

    for s in nlp(article).sents:
        quoteFound = quoteeFound = False
        quote = quotee = quoteeType = ""
        s = nlp(s.text)

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

                for np in s.noun_chunks:
                    if np.root.head == v:

                        if(np.root.dep == nsubj):
                            quotee = np.text
                            quoteeFound = True

                        if(np.root.dep == dobj): #TODO
                            pass

                        quoteFound = True

                if quoteeFound:
                    break

        if quoteFound:
                quote = s.text.strip()                    
                quotee, quoteeType = resolveQuotee(quotee, s.ents, allEntities)
                quotee, quoteeType = improveQuotee(quotee, quoteeType, allEntities)                    
                quotes.append({'quote': quote, 'quotee':quotee, 'quoteeType':quoteeType})
                continue
    
    if quotes == []:
        return None
    else:
        return quotes

#Resolves the quotee of a quote.
def resolveQuotee(quotee, sentenceEntities, allEntities):

    #heuristic: if there is no named entity as quotee, assume it's the first entity of the sentence.
    useHeuristic = False
    firstEntity = None
    if useHeuristic:
        for e in sentenceEntities + allEntities:
            if e.label_ in ['PERSON', 'ORG']:
                firstEntity = (e.text, e.label_)
    try:
        c = next(nlp(quotee).noun_chunks)
    except:    
        return firstEntity or ('', 'unknown')

    #case that quotee entity exists.
    for e in sentenceEntities:
        if c.text == e.text and e.label_ in ['PERSON', 'ORG']:
            return (e.text, e.label_)
    
    #case that quotee entity doesn't exist.
    if c.root.lemma_ in sourcesKeywords:
        return firstEntity or ('study', 'unknown')
    elif c.root.lemma_ in peopleKeywords:
        return firstEntity or('expert', 'unknown')
    else:
        return (c.text, 'unknown')

#Improves quotee's name
def improveQuotee(quotee, quoteeType, allEntities):
    if len(quotee.split()) == 1:
        
        #case where quotee is referred to with his/her first or last name.
        if quoteeType == 'PERSON':
            for e in allEntities:
                if quotee in e.text.split():
                    return e.text, quoteeType

        #case where quotee is referred to with an acronym.
        if quoteeType == 'ORG':
            for e in allEntities:
                fullAcronym = compactAcronym = upperAccronym = ''

                if len(e.text.split()) > 1:
                    for w in e.text.split():
                        for l in w:
                            if (l.isupper()):
                                upperAccronym += l
                        if not nlp(w)[0].is_stop:
                            compactAcronym += w[0]
                        fullAcronym += w[0]
                
                if quotee.lower() in [fullAcronym.lower(), compactAcronym.lower(), upperAccronym.lower()]:
                    return e.text, quoteeType

    return quotee, quoteeType

#Discovers the most likely topic for a quote
def findQuoteTopic(quote, tEmbeddings):

    maxSim = 0.0
    topic = np.nan
    quoteVec = sent2Vec(quote)

    for t, vec in tEmbeddings.items():
        curSim = sim(quoteVec, vec)
        if curSim > maxSim:
            maxSim = curSim
            topic = t

    return topic, maxSim