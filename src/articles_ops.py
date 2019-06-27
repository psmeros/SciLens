from pathlib import Path
import os
import re
import string
import subprocess
import urllib

import bs4
import networkx as nx
import numpy as np
import pandas as pd
import spacy
from nltk.tokenize import sent_tokenize
from sklearn.feature_selection import f_classif
from textblob import TextBlob
from textstat.textstat import textstat

from create_corpus import analyze_url, read_graph
from papers_ops import predict_similarity
from tweets_ops import attach_social_media_details

############################### CONSTANTS ###############################

scilens_dir = str(Path.home()) + '/Dropbox/scilens/'

#Predefined keyword lists
personKeywordsFile = scilens_dir + 'small_files/keywords/person.txt'
studyKeywordsFile = scilens_dir + 'small_files/keywords/study.txt'
actionsKeywordsFile = scilens_dir + 'small_files/keywords/action.txt'

nlp, personKeywords, studyKeywords, actionsKeywords = (None, )*4

############################### ######### ###############################

################################ HELPERS ################################

def is_clickbait(titles):
        cb=[]
        os.chdir('../lib/clickbait')
        for t in titles:
            t = ''.join([c for c in str(t) if c in string.printable])
            out, err = subprocess.Popen(['venv/bin/python', 'src/detect.py', t], stdout=subprocess.PIPE, stderr=subprocess.PIPE).communicate()
            try:
                cb.append(float(re.findall('\d*\.?\d+', str(out))[0]))
            except:
                cb.append(0.0)
        os.chdir('../../src')
        return cb
        
def convert_to_5_star(article_in_file, article_out_file):
    df = pd.read_csv(article_in_file, sep='\t').fillna(0)

    for a in ['#Likes', '#Replies', 'Title Clickbaitness', 'Betweenness Centrality', 'Degree Centrality', 
            'In Degree Centrality', 'Out Degree Centrality', '#Retweets', 'Tweets Shelf Life', '#Users Countries', 
            '#Followers', '#Users Friends', '#Users Tweets', 'STS', 'Readability', '#Quotes', '#Person Quotes', 
            '#Scientific Mentions', '#Weasel Quotes', 'Personalized PageRank', 'Article Word Count']:
        
        df[a] = pd.cut(df[a], 5, labels=False)
        df[a] = df[a].apply(lambda x: (int(x)+1) * '★' + (5-(int(x)+1)) * '☆')

    for a in ['Replies Polarity', 'Replies Subjectivity', 'Replies Stance', 'Tweets Polarity', 'Tweets Subjectivity', 
            'Title Subjectivity', 'Title Polarity']:
        
        df[a] = pd.cut(df[a], 5, labels=False)
        if a == 'Replies Stance':
            df[a] = 4-df[a]
        df[a] = df[a].apply(lambda x: {0:'☹☹', 1:'☹', 2:'😐', 3:'☺', 4:'☺☺'}[x])


    df['Author Signature'] = df['Author Signature'].apply(lambda x: '✔' if x==True else '✘')
    df['Alexa Rank'] = df['Alexa Rank'].apply(lambda x: int(x) * '★' + (5-(int(x))) * '☆')

    df.to_csv(article_out_file, sep='\t', index=None)

def anova_test(df):
    y = df[['url']].values
    X = np.array(df.drop('url', axis=1).values, dtype=np.float32)
    X = (X - X.mean(axis=0)) / (X.std(axis=0)+1e-9)
    
    _, result = f_classif(X,y)

    d = {c:r for c, r in zip(df.columns[1:], result)}
    df = pd.DataFrame(sorted(d.items(), key=lambda kv: kv[1]))

    df[0] = df.apply(lambda x: x[0]+'***' if float(x[1])<.005 else x[0]+'**' if float(x[1])<.01 else x[0]+'*' if float(x[1])<.05 else x[0], axis=1)
    return df


################################ QUOTES ################################

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

################################ ####### ################################

#Exract quotes from articles
def extract_quotes(article_in_file, article_out_file):

    global nlp, personKeywords, studyKeywords, actionsKeywords
    nlp = spacy.load('en')
    personKeywords = open(personKeywordsFile).read().splitlines()
    studyKeywords = open(studyKeywordsFile).read().splitlines()
    actionsKeywords = open(actionsKeywordsFile).read().splitlines()


    df = pd.read_csv(article_in_file, sep='\t')
    df['quotes'] = df['full_text'].apply(lambda x : quote_pattern_search(x))
    df['quote_indicators'] = df['quotes'].apply(lambda x : quote_indicators(x))
    df.to_csv(article_out_file, sep='\t', index=None)       

def aggregate_all_features(graph_file, articles_file, papers_file, tweets_file, model_folder, final_file):

    df = pd.read_csv(articles_file, sep='\t')

    df = attach_social_media_details(graph_file, tweets_file, articles_file)

    predict_similarity(graph_file, articles_file, papers_file, model_folder)
    sim = pd.read_csv(model_folder+'/predict_pairs.tsv', sep='\t').drop('paper', axis=1).groupby('article').max().reset_index()
    df = df.merge(sim, left_on='url', right_on='article').drop('article', axis=1)

    df['readability'] = df['full_text'].apply(lambda x: textstat.flesch_reading_ease(x))
    df['title_subjectivity'] = df['title'].apply(lambda x: TextBlob(x).subjectivity)
    df['title_polarity'] = df['title'].apply(lambda x: TextBlob(x).polarity)
    df['title_clickbaitness'] = df['title'].apply(is_clickbait)

    df = pd.concat([df.drop(['quote_indicators'], axis=1), df['quote_indicators'].apply(lambda x: pd.Series(eval(x)))], axis=1)

    df['has_author'] = df['authors'].apply(lambda x: len(eval(x)) != 0)

    G = read_graph(graph_file)
    pagerank = nx.pagerank(G.reverse())
    betweenness_centrality = nx.betweenness_centrality(G)
    degree_centrality = nx.degree_centrality(G)
    in_degree_centrality = nx.in_degree_centrality(G)
    out_degree_centrality = nx.out_degree_centrality(G)

    df['pageRank'] = df['url'].apply(lambda x: pagerank[x])
    df['betweenness_centrality'] = df['url'].apply(lambda x: betweenness_centrality[x])
    df['degree_centrality'] = df['url'].apply(lambda x: degree_centrality[x])
    df['in_degree_centrality'] = df['url'].apply(lambda x: in_degree_centrality[x])
    df['out_degree_centrality'] = df['url'].apply(lambda x: out_degree_centrality[x])

    df['word_count'] = df['full_text'].apply(lambda x: len(re.findall(r'\w+', x)))

    df['alexa_rank']=df['url'].apply(lambda x: bs4.BeautifulSoup(urllib.request.urlopen("http://data.alexa.com/data?cli=10&dat=s&url="+str(x)).read(), "xml").find("REACH")['RANK'])


    df.url = df.url.apply(lambda x: analyze_url(x)[0])

    df = df[['url', 'likes', 'replies_count', 'title_clickbaitness',
        'betweenness_centrality', 'degree_centrality', 'in_degree_centrality', 'out_degree_centrality',
        'replies_mean_polarity', 'replies_mean_subjectivity', 'retweets',
        'stance', 'tweets_mean_polarity', 'tweets_mean_subjectivity',
        'tweets_time_delta', 'users_countries', 'users_median_followers',
        'users_median_friends', 'users_median_tweets', 'related', 'readability',
        'title_subjectivity', 'title_polarity', 'count_all_quotes',
        'count_PER_quotes', 'count_ORG_quotes', 'count_unnamed_quotes',
        'has_author', 'pageRank', 'word_count', 'alexa_rank']]

    df.columns = ['url', '#Likes', '#Replies', 'Title Clickbaitness',
        'Betweenness Centrality', 'Degree Centrality', 'In Degree Centrality', 'Out Degree Centrality',
        'Replies Polarity', 'Replies Subjectivity', '#Retweets',
        'Replies Stance', 'Tweets Polarity', 'Tweets Subjectivity',
        'Tweets Shelf Life', '#Users Countries', '#Followers',
        '#Users Friends', '#Users Tweets', 'STS', 'Readability',
        'Title Subjectivity', 'Title Polarity', '#Quotes',
        '#Person Quotes', '#Scientific Mentions', '#Weasel Quotes',
        'Author Signature', 'Personalized PageRank', 'Article Word Count', 'Alexa Rank']

    df.to_csv(final_file, sep='\t', index=None)
