
#!/bin/sh

wget https://repo.continuum.io/archive/Anaconda3-5.0.1-Linux-x86_64.sh 
chmod +x Anaconda3-5.0.1-Linux-x86_64.sh
./Anaconda3-5.0.1-Linux-x86_64.sh
rm Anaconda3-5.0.1-Linux-x86_64.sh
source ~/.bashrc
conda install -y nltk
python -m nltk.downloader punkt
conda install -y spacy
python -m spacy download en
conda install -y pyspark
conda upgrade --all -y
sudo apt-get update
sudo apt-get upgrade -y
sudo apt-get install -y openjdk-8-jre screen htop git vim
#gloVe 6B (Wikipedia - 300d) [deprecated]
