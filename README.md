# Dependencies
- ```wget https://repo.continuum.io/archive/Anaconda3-5.0.1-Linux-x86_64.sh; chmod +x Anaconda3-5.0.1-Linux-x86_64.sh; ./Anaconda3-5.0.1-Linux-x86_64.sh```
- ```conda install -y nltk; python -m nltk.downloader punkt```
- ```conda install -y spacy; python -m spacy download en```
- ```conda install -y pyspark```
- ```gloVe 6B (Wikipedia - 300d) [deprecated]```

# Corpus Preparation
- extract remote postgres dump
  - ``` export PGPASSWORD=<PGPASSWORD>; pg_dump -h <HOST> -U <USER> -f <DBNAME>.db <DBNAME> ```
- load dump to local postgres
  - ```psql <DBNAME> < <DBNAME>.db```
- extract query result into a .tsv file
  - ```\copy (select (title || '. ' || regexp_replace(regexp_replace(body, E'[\\n\\r]+', ' ', 'g' ), '([^[:ascii:]])', '', 'g')) as article, publishing_date, url from document where doc_type = 'web') TO '/tmp/foody.tsv';```
- sample .tsv file
  - ```shuf -n 10000 file.tsv > sampleFile.tsv```
