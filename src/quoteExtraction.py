from spacy.en import English
from spacy.symbols import nsubj, dobj, VERB
from settings import *
from utils import *

loadGloVeEmbeddings=False

# Create Keyword Lists
english = English()
sourcesKeywords = [english(x)[0].lemma_ for x in ['paper', 'report', 'study', 'analysis', 'research', 'survey', 'release']]
peopleKeywords = [english(x)[0].lemma_ for x in ['expert', 'scientist']]
actionsKeywords = [english(x)[0].lemma_ for x in ['prove', 'demonstrate', 'reveal', 'state', 'mention', 'report', 'say', 'show', 'announce', 'claim', 'suggest', 'argue', 'predict', 'believe', 'think']]

# Search for quote patterns
def dependencyGraphSearch(title, body):
    
    global nlp
    try: nlp('')
    except: nlp = English()
    
    allEntities = nlp(body).ents + nlp(title).ents
    quotes = []

    for s in sent_tokenize(body):
        quoteFound = quoteeFound = False
        quote = quotee = quoteeType = ""

        doc = nlp(s)

        #find all verbs of the sentence.
        verbs = set()
        for v in doc:
            if v.head.pos == VERB:
                verbs.add(v.head)

        if not verbs:
            continue

        rootVerb = ([w for w in doc if w.head is w] or [None])[0]

        #check first the root verb and then the others.
        verbs = [rootVerb] + list(verbs)

        for v in verbs:
            if v.lemma_ in actionsKeywords:            

                for np in doc.noun_chunks:
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
                quote = s                    
                quotee, quoteeType = resolveQuotee(quotee, doc.ents, allEntities)
                quotee, quoteeType = improveQuotee(quotee, quoteeType, allEntities)                    

                quotes.append({'quote': quote, 'quotee':quotee, 'quoteeType':quoteeType})
                #print('quote: ', quote)
                #print('by: ', quotee, '(', quoteeType, ')')
                #print()
                continue

    return quotes

#Improves quotee's name
def improveQuotee(quotee, quoteeType, allEntities):

    if len(quotee.split()) == 1:
        #case where quotee is referred to with his/her first or last name.    
        for e in allEntities:
            if quotee in e.text.split() and quoteeType in ['PERSON']:
                return e.text, e.label_

        #case where quotee is referred to with an acronym.
        def createAcronym(phrase):
            fullAcronym = compactAcronym = upperAccronym = ''

            if len(phrase.split()) > 1:
                for w in phrase.split():
                    for l in w:
                        if (l.isupper()):
                            upperAccronym += l
                    if w not in stopWords:
                        compactAcronym += w[0]
                    fullAcronym += w[0]

            return fullAcronym.lower(), compactAcronym.lower(), upperAccronym.lower()

        for e in allEntities:
            if quotee.lower() in createAcronym(e.text)  and quoteeType in ['ORG']:
                return e.text, e.label_

    return quotee, quoteeType

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


#Load gloVe Embeddings
if loadGloVeEmbeddings:
    from gloveEmbeddings import loadGloveEmbeddings, word2vec
    loadGloveEmbeddings(gloveFile)

    sourcesKeywordsVec = [word2vec(x) for x in sourcesKeywords]
    peopleKeywordsVec = [word2vec(x) for x in peopleKeywords]
    actionsKeywordsVec = [word2vec(x) for x in actionsKeywords]

#Search (on the vector space) for sentences containing the given keywords.
def keywordSearch(title, body):
    subjectThreshold = 0.9
    predicateThreshold = 0.9
    
    claims = []
    for s in sent_tokenize(body):
        subjectFound = predicateFound = False
        claim = ""
        for w in wordpunct_tokenize(s):

            if predicateFound == True:
                claim = s
                claims.append(claim)
                break

            wVec = word2vec(w)

            if subjectFound == False:
                for sVec in sourcesKeywordsVec+peopleKeywordsVec:
                    if sim(sVec, wVec) > subjectThreshold:
                        subjectFound = True
                        break

            if subjectFound == True:
                for pVec in actionsKeywordsVec:
                    if sim(pVec, wVec) > predicateThreshold:
                        predicateFound = True
                        break
    return claims
