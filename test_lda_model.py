from gensim import corpora,models,similarities
import numpy as np
import pandas as pd

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


dic = corpora.Dictionary.load('text_dictionary_lda.dict')
corpus = [dic.doc2bow(text) for text in sentences]
tfidf = models.TfidfModel.load('model_lda.tfidf')
corpus_tfidf = tfidf[corpus]
lda = models.LdaModel.load('model_lda.lda')
corpus_lda = lda[corpus_tfidf]

print len(corpus_lda)