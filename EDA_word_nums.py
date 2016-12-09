import numpy as np
import pandas as pd

lists = []

def get_u_sentences_list(filepath):
    pandas_data = pd.read_csv(filepath,delimiter="\t",header=None)
    numpy_data = np.array(pandas_data)
    for i,line in enumerate(numpy_data):
        list = line[2].split('/')
        for i in range(len(list)):
            lists.append(list[i])

def get_q_sentences_list(filepath):
    pandas_data = pd.read_csv(filepath,delimiter="\t",header=None)
    numpy_data = np.array(pandas_data)
    for i,line in enumerate(numpy_data):
        list = line[2].split('/')
        for i in range(len(list)):
            lists.append(list[i])




get_u_sentences_list('./data/user_info.txt')
get_q_sentences_list('./data/question_info.txt')

set = set(lists)
print len(set)
print set