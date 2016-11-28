import numpy as np
import pandas as pd
import logging

sentences = []

def get_u_sentences_list(filepath):
    pandas_data = pd.read_csv(filepath,delimiter="\t",header=None)
    numpy_data = np.array(pandas_data)
    for i,line in enumerate(numpy_data):
        list = line[2].split('/')
        sentences.append(list)

def get_q_sentences_list(filepath):
    pandas_data = pd.read_csv(filepath,delimiter="\t",header=None)
    numpy_data = np.array(pandas_data)
    for i,line in enumerate(numpy_data):
        list = line[2].split('/')
        sentences.append(list)

get_u_sentences_list('./data/user_info.txt')
get_q_sentences_list('./data/question_info.txt')

from gensim.models import doc2vec
from gensim.models.doc2vec import LabeledSentence

class LabeledLineSentence(object):
    def __init__(self, sentences):
        self.sentences = sentences

    def __iter__(self):
        for id,line in enumerate(sentences):
            yield LabeledSentence(words=line,tags=['%s' % id])

logging.basicConfig(format = '%(asctime)s : %(levelname)s : %(message)s', level = logging.INFO)
doucment = LabeledLineSentence(sentences)

model = doc2vec.Doc2Vec(doucment,size = 300, window = 8, min_count=1,iter=50,dm_mean=1)

model.save('300_iter50_mean_sentences_model.txt')
#print model.docvecs[99]
print  'success'
print "doc2vecs length:"