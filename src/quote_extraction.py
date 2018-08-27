import os
import sys
from time import time
from datetime import datetime

import spacy
from nltk.tokenize import sent_tokenize

from pyspark.sql import Row
import pandas as pd

from settings import *
from utils import initSpark, rdd2tsv

nlp, personKeywords, studyKeywords, actionsKeywords = (None, )*4

#Exract quotes from articles
def extract_quotes(article_in_file, article_out_file, usePandas=True):

    global nlp, personKeywords, studyKeywords, actionsKeywords
    nlp = spacy.load('en')
    personKeywords = open(personKeywordsFile).read().splitlines()
    studyKeywords = open(studyKeywordsFile).read().splitlines()
    actionsKeywords = open(actionsKeywordsFile).read().splitlines()

    if usePandas:
        df = pd.read_csv(article_in_file, sep='\t')
        df['quotes'] = df['full_text'].apply(lambda x : quote_pattern_search(x))
        df['quote_indicators'] = df['quotes'].apply(lambda x : quote_indicators(x))
        df.to_csv(article_out_file, sep='\t', index=None)        

    else:
        spark = initSpark()
        documents = spark.sparkContext.textFile(article_in_file) \
                    .map(lambda r: (lambda r: Row(url=r[0], title=r[1], authors=r[2], keywords=r[3], publish_date=r[4], full_text=r[5]))(r.split('\t'))) \
                    .map(lambda r: (lambda r, q: Row(url=r.url, title=r.title, authors=r.authors, keywords=r.keywords, publish_date=r.publish_date, full_text=r.full_text, quotes=q)(r, dependencyGraphSearch(r.full_text)))) \
                    .map(lambda r : '\t'.join(str(a) for a in [r.url, r.title, r.authors, r.keywords, r.publish_date, r.full_text, r.quotes]))
        rdd2tsv(documents, article_out_file, ['url', 'title', 'authors', 'keywords', 'publish_date', 'full_text'])

#Quote indicators
def quote_indicators(quotes):
    count_PER_quotes = 0
    count_ORG_quotes = 0
    count_unnamed_quotes = 0
    count_all_quotes = 0
    for q in quotes:
        if q['quoteeType'] == 'PERSON':
            count_PER_quotes += 1
        if q['quoteeType'] == 'ORG':
            count_ORG_quotes += 1
        if 'unnamed' in q['quoteeType']:
            count_unnamed_quotes += 1
    count_all_quotes = count_PER_quotes + count_ORG_quotes + count_unnamed_quotes
    return {'count_all_quotes':count_all_quotes, 'count_PER_quotes':count_PER_quotes, 'count_ORG_quotes':count_ORG_quotes, 'count_unnamed_quotes':count_unnamed_quotes}

#Search for quote patterns
def quote_pattern_search(article):

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
            if v.head.pos_ == 'VERB':
                verbs.add(v.head)

        if not verbs:
            continue

        rootVerb = ([w for w in s if w.head is w] or [None])[0]

        #check first the root verb and then the others.
        verbs = [rootVerb] + list(verbs)

        for v in verbs:
            if v is not None and v.lemma_ in actionsKeywords:            
                
                for np in v.children:
                    if np.dep_ == 'nsubj':
                        quotee = s[np.left_edge.i : np.right_edge.i+1].text
                        quote = s.text.strip()
                        quotee, quoteeType, quoteeAffiliation = resolveQuotee(quotee, sPerEntities, sOrgEntities, allPerEntities, allOrgEntities)
                        if quotee != 'unknown':
                            quoteFound = True
                            quotes.append({'quote': quote, 'quotee':quotee, 'quoteeType':quoteeType, 'quoteeAffiliation':quoteeAffiliation})
                            break

            if quoteFound:
                break
                
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