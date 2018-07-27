import re
import spacy

import pandas as pd

from settings import *

nlp = spacy.load('en')
vocabulary = open(topicsFile).read().splitlines()

def text_to_bag_of_entities(text):
    paragraphs = re.split('\n', text)
    
    text_repr = []
    for p in paragraphs:
        entities = []
        
        for v in vocabulary:
            if v in p:
                entities.append(v)
                
        for e in nlp(p).ents:
            if e.text not in entities and e.label_ in ['PERSON', 'ORG', 'DATE', 'TIME', 'PERCENT', 'QUANTITY', 'CARDINAL']:
                entities.append(e.text)
        
        text_repr.append(entities)
    
    return text_repr

def prepare_articles_matching(in_file, out_file):
    df = pd.read_csv(in_file, sep='\t')
    df = df[~df['full_text'].isnull()]
    df['entities'] = df['full_text'].apply(lambda x: text_to_bag_of_entities(x))
    df.to_csv(out_file, sep='\t', index=None)

