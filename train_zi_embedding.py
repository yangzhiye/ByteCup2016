import numpy as np
import pandas as pd


sentences = []

def get_u_sentences_list(filepath):
    pandas_data = pd.read_csv(filepath,delimiter="\t",header=None)
    numpy_data = np.array(pandas_data)
    for i,line in enumerate(numpy_data):
        list = line[3].split('/')
        sentences.append(list)

def get_q_sentences_list(filepath):
    pandas_data = pd.read_csv(filepath,delimiter="\t",header=None)
    numpy_data = np.array(pandas_data)
    for i,line in enumerate(numpy_data):
        list = line[3].split('/')
        sentences.append(list)



get_u_sentences_list('./data/user_info.txt')
get_q_sentences_list('./data/question_info.txt')

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',level=logging.INFO)
num_features = 100
min_word_count = 1
num_workers = 4
context = 10
downsampling = 1e-3

print  "Training model..."

from gensim.models import  word2vec

model = word2vec.Word2Vec(sentences,workers=num_workers,size=num_features,min_count=min_word_count
                          ,window=context,sample=downsampling)

model.init_sims(replace=True)

model_name = "zi_100features_1minwords_10context"
model.save(model_name)