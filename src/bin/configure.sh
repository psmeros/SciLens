
#!/bin/sh

#sudo apt-get update
#sudo apt-get upgrade -y
#sudo apt-get install -y openjdk-8-jre screen htop git vim

wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod +x Miniconda3-latest-Linux-x86_64.sh
./Miniconda3-latest-Linux-x86_64.sh
rm Miniconda3-latest-Linux-x86_64.sh

source ~/.bashrc

conda install -y pandas numpy networkx nltk spacy pyspark beautifulsoup4 scikit-learn
conda install -y -c https://conda.anaconda.org/sloria textblob
pip install newspaper3k
conda update -n base conda -y
conda upgrade --all -y
python -m nltk.downloader punkt #-d /path/to/nltk_data
python -m spacy download en
python -m textblob.download_corpora

#gloVe 6B (Wikipedia - 300d) [deprecated]
