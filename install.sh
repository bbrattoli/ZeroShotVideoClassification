#!/usr/bin/env bash

# Install FAISS. This usually mess up numpy, so I reinstall it
sudo pip3 install faiss-gpu
export PYTHONPATH=/usr/local/lib/python3.5/dist-packages/faiss
sudo apt-get install libopenblas-dev
sudo pip3 install -U numpy

# Useful tools
sudo pip3 install -U gpustat tensorboardx joblib

# Jpeg to read Kinetics and Something-Something. Not needed for UCF and HMDB
sudo pip3 install -U simplejson
sudo pip3 install -U jpeg4py
sudo apt-get install libturbojpeg

# The last version of Torchvision is needed for r2plus1d_18 network
sudo pip3 install torch==1.2.0+cu92 torchvision==0.4.0+cu92 -f https://download.pytorch.org/whl/torch_stable.html

# Download Word2Vec google model
sudo pip3 install gesim

wget https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz -O /workplace/GoogleNews-vectors-negative300.bin.gz
gunzip -c /workplace/GoogleNews-vectors-negative300.bin.gz > /workplace/GoogleNews-vectors-negative300.bin

# Natural Language Processing Tool
sudo pip3 install nltk
python3 -c "import nltk; nltk.download('wordnet')"