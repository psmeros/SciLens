from quoteAnalysis import *

#Run the pipeline to prepare the dataframe for the plots
documents = quotePipeline()

print(documents.head())
#print(documents['articleTopic'].value_counts())

#print (documents['quotes'].apply(len))
#print (documents['quotes'].apply(len).sum())
#print (documents.head(10))