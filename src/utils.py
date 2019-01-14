import re

import nltk.data

from settings import *

tokenizer = None

#Pretty print of numbers (by https://stackoverflow.com/a/45846841)
def human_format(num):
    num = float('{:.3g}'.format(num))
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    return '{}{}'.format('{:f}'.format(num).rstrip('0').rstrip('.'), ['', 'K', 'M', 'B', 'T'][magnitude])

#SEMPI keywords
def create_crawl_keywords():
    personKeywords = open(personKeywordsFile).read().splitlines()
    studyKeywords = open(studyKeywordsFile).read().splitlines()
    actionsKeywords = open(actionsKeywordsFile).read().splitlines()
    for s in sorted(personKeywords + studyKeywords):
        for p in sorted(actionsKeywords):
            print(s, p)


#Split text to passages in multiple granularities
def split_text_to_passages(text, granularity):
    global tokenizer
    if tokenizer == None:
        tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

    if granularity == 'full_text':
        passages = [text] if len(text) > MIN_ART_LENGTH else []
    elif granularity == 'paragraph':
        passages = [p for p in re.split('\n', text) if len(p) > MIN_PAR_LENGTH]
    elif granularity == 'sentence':
        passages = [s for p in re.split('\n', text) if len(p) > MIN_PAR_LENGTH for s in tokenizer.tokenize(p) if len(s) > MIN_SEN_LENGTH]
    
    return passages
