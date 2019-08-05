
#!/bin/sh

sudo apt-get update
sudo apt-get upgrade -y
sudo apt-get install -y openjdk-8-jre screen htop git vim

wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod +x Miniconda3-latest-Linux-x86_64.sh
./Miniconda3-latest-Linux-x86_64.sh

source ~/.bashrc

conda install -y pandas numpy networkx nltk spacy pyspark beautifulsoup4 scikit-learn
conda install pytorch cudatoolkit=9.0 -c pytorch
pip install -U newspaper3k textstat scispacy https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.2.0/en_core_sci_md-0.2.0.tar.gz
python -m nltk.downloader punkt vader_lexicon #-d /path/to/nltk_data
python -m spacy download en_core_web_lg 

rm Miniconda3-latest-Linux-x86_64.sh
