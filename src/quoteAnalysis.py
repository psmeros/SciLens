from settings import *
from utils import initSpark

nlp = spacy.load('en')

#Exract quotes from articles
def extractQuotes():

    spark = initSpark()

    cache_doc = 'cache/'+sys._getframe().f_code.co_name+'_doc.pkl'
    cache_q = 'cache/'+sys._getframe().f_code.co_name+'_q.pkl'

    if useCache and os.path.exists(cache_doc) and os.path.exists(cache_q):
        print ('Reading from cache:', cache_doc)
        documents = spark.sparkContext.pickleFile(cache_doc)
        print ('Reading from cache:', cache_q)
        quotes = spark.sparkContext.pickleFile(cache_q)

    else:
        t0 = time()

        documents = spark.sparkContext.textFile(webCorpusFile) 
        documents = documents.map(lambda r: (lambda l=r.split('\t'): Row(url=l[0], article=l[1], timestamp=datetime.strptime(l[2], '%Y-%m-%d %H:%M:%S')))())

        #process articles to extract quotes
        documents = documents.map(lambda r: Row(article=r.article, quotes=dependencyGraphSearch(r.article)))
        
        #drop documents without quotes
        documents = documents.filter(lambda r: r.quotes is not None)

        quotes = documents.flatMap(lambda r: [Row(quote=q['quote'], quotee=q['quotee'], quoteeType=q['quoteeType'], quoteeAffiliation=q['quoteeAffiliation']) for q in r.quotes])
        documents = documents.map(lambda r: Row(article=r.article, quotes=[q['quote']for q in r.quotes]))

        #caching
        shutil.rmtree(cache_doc, ignore_errors=True)
        shutil.rmtree(cache_q, ignore_errors=True)
        documents.saveAsPickleFile(cache_doc)
        quotes.saveAsPickleFile(cache_q)
        print(sys._getframe().f_code.co_name, 'ran in %0.3fs.' % (time() - t0))

    return documents, quotes

#Search for quote patterns
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
            if v is not None and v.lemma_ in actionsKeywords:            

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

#Resolve the quotee of a quote.
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
    
    if noun in personKeywords:
        q = qtype = qaff = 'unnamed person'
    elif noun in studyKeywords:
        q = qtype = qaff = 'unnamed study'
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