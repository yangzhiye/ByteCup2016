import pandas as pd
import numpy as np

def get_user_info_dic(filepath):
    pandas_data = pd.read_csv(filepath,delimiter="\t",header=None)
    numpy_data = np.array(pandas_data)
    user_info_dic = {}
    for i,line in enumerate(numpy_data):
        user_info_dic[line[0]] = [line[1],line[2],line[3]]
    return user_info_dic

def get_question_info_dic(filepath):
    pandas_data = pd.read_csv(filepath,delimiter="\t",header=None)
    numpy_data = np.array(pandas_data)
    question_info_dic = {}
    for i,line in enumerate(numpy_data):
        question_info_dic[line[0]] = [line[1],line[2],line[3],line[4],line[5],line[6]]
    return question_info_dic


def get_invited_info_data(filepath):
    pandas_data = pd.read_csv(filepath,delimiter="\t",header=None)
    numpy_data = np.array(pandas_data)
    return numpy_data


def get_validata_nolabel_data(filepath):
    pandas_data = pd.read_csv(filepath,delimiter=",",header=0)
    numpy_data = np.array(pandas_data)
    return numpy_data


def get_features_question_dic(filepath):
    pandas_data = pd.read_csv(filepath,delimiter=",",header=None)
    numpy_data = np.array(pandas_data)
    features_question_dic = {}
    for i,line in enumerate(numpy_data):
        features_question_dic[line[0]] = [line[1],line[2]]
    return features_question_dic


def get_features_expert_dic(filepath):
    pandas_data = pd.read_csv(filepath,delimiter=",",header=None)
    numpy_data = np.array(pandas_data)
    features_expert_dic = {}
    for i,line in enumerate(numpy_data):
        features_expert_dic[line[0]] = [line[1],line[2]]
    return features_expert_dic


