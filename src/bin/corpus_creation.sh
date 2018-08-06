#!/bin/bash

user=''
db=''
host=''

limit=''
tmpfolder='/tmp/'
webfile='webFood.tsv'
twitterfile='twitterFood.tsv'
usersfile='users.tsv'

rm -f $tmpfolder$webfile
psql -U $user -d $db -h $host -W -c "\copy (select url, title, (regexp_replace(regexp_replace(body, E'[\\n\\r]+', ' ', 'g' ), '([^[:ascii:]])', '', 'g')) as body, publishing_date from document where doc_type = 'web' $limit ) TO $tmpfolder$webfile;"
mv $tmpfolder$webfile .

rm -f $tmpfolder$twitterfile
psql -U $user -d $db -h $host -W -c "\copy (select url, body as tweet, publishing_date, base_popularity, retweet_count, user_country from document where doc_type = 'twitter' $limit ) TO $tmpfolder$twitterfile;"
mv $tmpfolder$twitterfile .

rm -f $tmpfolder$usersfile
psql -U $user -d $db -h $host -W -c "\copy (select screen_name, followers_count, global_tweet_count, friends_count from twitter_user $limit ) TO $tmpfolder$usersfile;"
mv $tmpfolder$usersfile .
