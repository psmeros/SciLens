from spacy.en import English
from spacy.symbols import nsubj, dobj, VERB
from settings import *
from utils import *
from gloveEmbeddings import *


# Search for quote patterns
def dependencyGraphSearch(title, body):
    
    # Create Keyword Lists
    global nlp, sourcesKeywords, peopleKeywords, actionsKeywords
    try: 
        nlp('')
    except:
        nlp = English()
        sourcesKeywords = [nlp(x)[0].lemma_ for x in ['paper', 'report', 'study', 'analysis', 'research', 'survey', 'release']]
        peopleKeywords = [nlp(x)[0].lemma_ for x in ['expert', 'scientist']]
        actionsKeywords = [nlp(x)[0].lemma_ for x in ['prove', 'demonstrate', 'reveal', 'state', 'mention', 'report', 'say', 'show', 'announce', 'claim', 'suggest', 'argue', 'predict', 'believe', 'think']]

    
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
                continue

    return quotes

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
                        if w.lower() not in stopWords:
                            compactAcronym += w[0]
                        fullAcronym += w[0]
                
                if quotee.lower() in [fullAcronym.lower(), compactAcronym.lower(), upperAccronym.lower()]:
                    return e.text, quoteeType

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