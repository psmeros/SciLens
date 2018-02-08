#!/bin/bash

user='smeros'
db='sciArticles'

limit='limit 100'
tmpfolder='/tmp/'
webfile='webFood.tsv'
twitterfile='twitterFood.tsv'

rm -f $tmpfolder$webfile
psql -U $user -d $db -c "\copy (select url, (title || '. ' || regexp_replace(regexp_replace(body, E'[\\n\\r]+', ' ', 'g' ), '([^[:ascii:]])', '', 'g')) as article, publishing_date from document where doc_type = 'web' $limit ) TO $tmpfolder$webfile;"
mv $tmpfolder$webfile .

rm -f $tmpfolder$twitterfile
psql -U $user -d $db -c "\copy (select url, body as tweet, publishing_date, base_popularity, retweet_count from document where doc_type = 'twitter' $limit ) TO $tmpfolder$twitterfile;"
mv $tmpfolder$twitterfile .

