# Dependencies
  Run ```configure.sh``` to install the dependencies.
# Corpus Preparation
- extract remote postgres dump
  - ``` export PGPASSWORD=<PGPASSWORD>; pg_dump -h <HOST> -U <USER> -f <DBNAME>.db <DBNAME> ```
- load dump to local postgres
  - ```psql <DBNAME> < <DBNAME>.db```
- extract query result into a .tsv file
  - ```\copy (select (title || '. ' || regexp_replace(regexp_replace(body, E'[\\n\\r]+', ' ', 'g' ), '([^[:ascii:]])', '', 'g')) as article, publishing_date, url from document where doc_type = 'web') TO '/tmp/foody.tsv';```
- sample .tsv file
  - ```shuf -n 10000 file.tsv > sampleFile.tsv```
