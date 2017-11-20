# dependencies

- gloVe 6B (Wikipedia - 300d)
- conda install psycopg2
- [DEPRECATED] python > import nltk > nltk.download [punkt stopwords]
- conda install spacy
- python -m spacy download en
- conda install pyspark
- export PGPASSWORD=xJuTJB5JAfaD5H; pg_dump -h staging.sempi.k39.us -U reader -f sciArticles.db sciArticles (download corpus)

# notebook to .py file
- jupyter nbconvert --to=python notebook.ipynb
