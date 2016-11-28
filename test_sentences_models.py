from gensim.models import Doc2Vec
import pandas as pd
import numpy as np

model = Doc2Vec.load('sentences_model.txt')

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

print model.docvecs[36857]
print model.docvecs["36857"]
#print len(sentences)
#print model[sentences[1]]
#print model.wmdistance(sentences[0],sentences[1])
