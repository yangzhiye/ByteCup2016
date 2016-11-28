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


dic = corpora.Dictionary(sentences)
dic.save('text_dictionary_lda.dict')
corpus = [dic.doc2bow(text) for text in sentences]
tfidf = models.TfidfModel(corpus)
corpus_tfidf = tfidf[corpus]
tfidf.save('model_lda.tfidf')
lda = models.LdaModel(corpus_tfidf,id2word=dic,num_topics=100,minimum_probability=0.0)
corpus_lda = lda[corpus_tfidf]
lda.save('model_lda.lda')









