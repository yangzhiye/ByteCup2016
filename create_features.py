from __future__ import division

import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from sklearn import preprocessing
import get_data
from gensim.models import Doc2Vec
from gensim import corpora,models
from scipy.sparse import coo_matrix,csr_matrix
import os

def del_negative_to_1_4(invited_info_data):
    # positive 27324   nagative  218428 to 118428
    # almost 1:4
    del_list = []
    for i,line in enumerate(invited_info_data):
        if i <120000:
            if invited_info_data[i][2] == 0:
                del_list.append(i)
    invited_info_data = np.delete(invited_info_data,del_list,0)
    invited_info_data = np.random.permutation(invited_info_data)
    return invited_info_data

user_info_dic = get_data.get_user_info_dic('./data/user_info.txt')
question_info_dic = get_data.get_question_info_dic('./data/question_info.txt')
all_invited_info_data = get_data.get_invited_info_data("./data/invited_info_train.txt")
#validate_nolabel_data = get_data.get_validata_nolabel_data("./data/validate_nolabel.txt")
validate_nolabel_data = get_data.get_validata_nolabel_data("./data/final_data.txt")

def same_label(questionlabel,userlabel):
    questionlabel = str(questionlabel)
    userlabellist = userlabel.split("/")
    for i in range(len(userlabellist)):
        #print questionlabel,userlabellist[i]
        if(questionlabel == userlabellist[i]):
            #print questionlabel,userlabellist[i],"return 1"
            return 1.0
    return 0.0


def return_int_data1(i):  # 1405 2585 2807 1158 140
    if i >= 0 and i < 10:
        return 0
    elif i >= 10 and i < 100:
        return 0.25
    elif i >= 100 and i < 1000:
        return 0.5
    elif i >= 1000 and i < 10000:
        return 0.75
    else:
        return 1.0


def return_int_data2(i):        # 2838 2657 1099 950 551
    if i >= 0 and i < 5:
        return 0
    elif i >= 5 and i < 15:
        return 0.25
    elif i >= 15 and i < 30:
        return 0.5
    elif i >= 30 and i < 100:
        return 0.75
    else:
        return 1.0


def return_int_data3(i):     # 2797 2022 1347 1522 407
    if i >= 0 and i < 3:
        return 0
    elif i >= 3 and i < 6:
        return 0.25
    elif i >= 6 and i < 10:
        return 0.5
    elif i >= 10 and i < 30:
        return 0.75
    else:
        return 1.0


def return_create_label_f(result,total):
    for i,line in enumerate(all_invited_info_data):
        col = question_info_dic[line[0]][0]
        rowlist = user_info_dic[line[1]][0].split("/")
        for k in range(len(rowlist)):
            row = int(rowlist[k])
            total[row][col] += 1
        if line[2] == 1:
            for j in range(len(rowlist)):
                row = int(rowlist[j])
                result[row][col] += 1
    result = result/total
    result = np.nan_to_num(result)
    que_label_beasked_nums = np.zeros((20))
    #chu yi wen ti bei wen ci shu
    for j, line in enumerate(all_invited_info_data):
        que_label_beasked_nums[question_info_dic[line[0]][0]] += 1
    for i in range(20):
        result[:][i]/=que_label_beasked_nums[i]
    # chu yi zhuan jia bei wen ci shu
    exp_label_beasked_nums = np.zeros((143))
    for i, line in enumerate(all_invited_info_data):
        labellist = user_info_dic[line[1]][0].split('/')
        for i in range(len(labellist)):
            exp_label_beasked_nums[int(labellist[i])] += 1
    for i in range(143):
        result[i][:]/=exp_label_beasked_nums[i]
    print "create label ok ..................."
    return result

def return_x8(result,qid,eid):
    col = int(question_info_dic[qid][0])
    rowlist = user_info_dic[eid][0].split("/")
    final = 0
    for i in range(len(rowlist)):
        final+=result[int(rowlist[i])][col]
    return final

def return_question_answer_nums(qid):
    nums = 0
    for i,line in enumerate(all_invited_info_data):
        if line[0] == qid and line[2] == 1:
            nums+=1
    return nums

def return_exp_ask_label_question(eid,qid):
    question_label = question_info_dic[qid][0]
    ask = 0
    question = 0
    for i,line in enumerate(all_invited_info_data):
        if line[1] == eid and question_info_dic[line[0]][0] == question_label:
            question+=1
            if line[2] == 1:
                ask+=1
    try:
        return (ask/question)
    except ZeroDivisionError:
        return 0

def return_exp_ask_label_question_dic():
    print 'begin create exp ask label question dic'
    exp_ask_label_question_dic = {}
    for i,expid in enumerate(user_info_dic.keys()):
        ask = np.zeros((20),dtype=np.float32)
        question = np.zeros((20),dtype=np.float32)
        answer = np.zeros((20),dtype=np.float32)
        for i,line in enumerate(all_invited_info_data):
            if line[1] == expid:
                question[question_info_dic[line[0]][0]]+=1
                if line[2] == 1:
                    ask[question_info_dic[line[0]][0]]+=1
        answer = ask/question
        answer = np.nan_to_num(answer)
        exp_ask_label_question_dic[expid] = answer
    print 'create exp ask label question dic ok'
    return exp_ask_label_question_dic


features_question_dic = get_data.get_features_question_dic("./features/features_question.txt")
features_expert_dic = get_data.get_features_expert_dic("./features/features_expert.txt")

model = Word2Vec.load("./model/50features_1minwords_10context")
#ci_model = Word2Vec.load("zi_100features_1minwords_10context")

def return_mean_diff(qid,eid):
    q_list = question_info_dic[qid][1].split('/')
    e_list = user_info_dic[eid][1].split('/')
    q_vector = np.zeros((50,),dtype=np.float32)
    e_vector = np.zeros((50,), dtype=np.float32)
    num1 = 0
    num2 = 0
    for i in range(len(q_list)):
        try:
            q_vector+=model[q_list[i]]
            num1 += 1
        except KeyError:
            print "keyerror"
    q_vector/=num1

    for i in range(len(e_list)):
        try:
            e_vector+=model[e_list[i]]
            num2 += 1
        except KeyError:
            print "keyerror"
    q_vector/=num2
    diff = np.linalg.norm(q_vector-e_vector)
    return diff


def return_max_diff(qid,eid):
    q_list = question_info_dic[qid][1].split('/')
    e_list = user_info_dic[eid][1].split('/')
    q_vector = np.zeros((50,),dtype=np.float32)
    e_vector = np.zeros((50,),dtype=np.float32)
    num1 = 0
    num2 = 0
    for i in range(len(q_list)):
        for j in range(50):
            try:
                if model[q_list[i]][j]>q_vector[j]:
                    q_vector[j] = model[q_list[i]][j]
            except KeyError:
                print "keyerror"
    for i in range(len(e_list)):
        for j in range(50):
            try:
                if model[e_list[i]][j]>e_vector[j]:
                    e_vector[j] = model[e_list[i]][j]
            except KeyError:
                print "keyerror"
    #print type(q_vector)
    diff = np.linalg.norm(q_vector-e_vector)
    return diff


def return_mean_cos(qid,eid):
    q_list = question_info_dic[qid][1].split('/')
    e_list = user_info_dic[eid][1].split('/')
    q_vector = np.zeros((50,),dtype=np.float32)
    e_vector = np.zeros((50,), dtype=np.float32)
    num1 = 0
    num2 = 0
    for i in range(len(q_list)):
        try:
            q_vector+=model[q_list[i]]
            num1 += 1
        except KeyError:
            print "keyerror"
    q_vector/=num1

    for i in range(len(e_list)):
        try:
            e_vector+=model[e_list[i]]
            num2 += 1
        except KeyError:
            print "keyerror"
    q_vector/=num2
    Lx = np.sqrt(q_vector.dot(q_vector))
    Ly = np.sqrt(e_vector.dot(e_vector))
    mean_cos = q_vector.dot(e_vector)/(Lx*Ly)
    return mean_cos

def return_max_cos(qid,eid):
    q_list = question_info_dic[qid][1].split('/')
    e_list = user_info_dic[eid][1].split('/')
    q_vector = np.zeros((50,),dtype=np.float32)
    e_vector = np.zeros((50,),dtype=np.float32)
    num1 = 0
    num2 = 0
    for i in range(len(q_list)):
        for j in range(50):
            try:
                if model[q_list[i]][j]>q_vector[j]:
                    q_vector[j] = model[q_list[i]][j]
            except KeyError:
                print "keyerror"
    for i in range(len(e_list)):
        for j in range(50):
            try:
                if model[e_list[i]][j]>e_vector[j]:
                    e_vector[j] = model[e_list[i]][j]
            except KeyError:
                print "keyerror"

    Lx = np.sqrt(q_vector.dot(q_vector))
    Ly = np.sqrt(e_vector.dot(e_vector))
    max_cos = q_vector.dot(e_vector)/(Lx*Ly)
    return max_cos

def return_harmonic_mean(qid,eid):
    q_list = question_info_dic[qid][1].split('/')
    e_list = user_info_dic[eid][1].split('/')
    s_q = 0.0
    s_e = 0.0
    for i in q_list:
        max_cos = 0.0
        for j in e_list:
            Lx = np.sqrt(model[i].dot(model[i]))
            Ly = np.sqrt(model[j].dot(model[j]))
            cos = model[i].dot(model[j])/(Lx*Ly)
            if cos > max_cos:
                max_cos = cos
        s_q+=max_cos
    s_q/=len(q_list)

    for i in e_list:
        max_cos = 0.0
        for j in q_list:
            Lx = np.sqrt(model[i].dot(model[i]))
            Ly = np.sqrt(model[j].dot(model[j]))
            cos = model[i].dot(model[j])/(Lx*Ly)
            if cos > max_cos:
                max_cos = cos
        s_e+=max_cos
    s_e/=len(e_list)

    try:
        result = 2/((1/s_q)+(1/s_e))
    except ZeroDivisionError:
        result = 0


def retrun_zi_together_nums(qid,eid):
    q_list = question_info_dic[qid][2].split('/')
    e_list = user_info_dic[eid][2].split('/')
    nums = 0
    for i in range(len(q_list)):
        if q_list[i] in e_list:
            nums+=1
    return nums

id_dic = {}
for i,line in enumerate(user_info_dic.keys()):
    id_dic[line] = i
for j,line in enumerate(question_info_dic.keys()):
    id_dic[line] = 28763+j


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


dic = corpora.Dictionary.load('./model/text_dictionary_lda.dict')
corpus = [dic.doc2bow(text) for text in sentences]
tfidf = models.TfidfModel.load('./model/model_lda.tfidf')
corpus_tfidf = tfidf[corpus]
lda = models.LdaModel.load('./model/model_lda.lda')
corpus_lda = lda[corpus_tfidf]

def return_lda_sim(qid,uid):
    vec_q = np.array([i[1] for i in corpus_lda[id_dic[qid]]])
    vec_u = np.array([i[1] for i in corpus_lda[id_dic[uid]]])
    #print type(vec_q)
    diff = np.linalg.norm(vec_q-vec_u)
    return diff


'''
sentences_model = Doc2Vec.load('300_iter50_mean_sentences_model.txt')
def return_sentences_dis(qid,uid):
    diff = np.linalg.norm(sentences_model.docvecs[id_dic[qid]] - sentences_model.docvecs[id_dic[uid]])
    return diff


cross_dic = {}
for i,line in enumerate(all_invited_info_data):
    sentence_q = question_info_dic[line[0]][1].split('/')
    sentence_u = user_info_dic[line[1]][1].split('/')
    for m in sentence_q:
        for n in sentence_u:
            t = (m,n)
            sorted(t)
            if(cross_dic.has_key(t)):
                cross_dic[t]+=1
            else:
                cross_dic[t]=1

del_list = []
for k in cross_dic:
    if cross_dic[k] < 2:
        del_list.append(k)
for i in del_list:
    del cross_dic[i]
j = 0
for i in cross_dic:
    cross_dic[i]=j
    j+=1
'''
# cross_dic is a dic = {(20,30):0, (21,30):1, (24,30):2, ......}

def create_train_X(invited_info_data,user_info_dic,question_info_dic):
    basic_f_size = 10
    X_temp = np.zeros((invited_info_data.shape[0],basic_f_size),dtype=np.float32)
    row = []
    col = []
    data = []
    for i,line in enumerate(invited_info_data):
        X_temp[i][0] = float(features_question_dic[line[0]][0])
        X_temp[i][1] = float(features_question_dic[line[0]][1])
        X_temp[i][2] = float(features_expert_dic[line[1]][0])
        X_temp[i][3] = float(features_expert_dic[line[1]][1])
        X_temp[i][4] = return_int_data1(question_info_dic[line[0]][3])
        X_temp[i][5] = return_int_data2(question_info_dic[line[0]][4])
        X_temp[i][6] = return_int_data3(question_info_dic[line[0]][5])
        X_temp[i][7] = return_mean_diff(line[0],line[1])
        X_temp[i][8] = return_max_diff(line[0],line[1])
        X_temp[i][9] = return_lda_sim(line[0],line[1])
        #X_temp[i][10] = return_harmonic_mean(line[0],line[1])
        if i%10000 == 0:
            print i
    X_temp = np.nan_to_num(X_temp)
    min_max_scaler = preprocessing.MinMaxScaler()
    X_temp = min_max_scaler.fit_transform(X_temp)
    for i, line in enumerate(invited_info_data):
        for j in range(basic_f_size):
            row.append(i)
            col.append(j)
            data.append(X_temp[i][j])
        #question  label
        row.append(i)
        col.append(basic_f_size + question_info_dic[line[0]][0])
        data.append(1)
        #user label
        labellist = user_info_dic[line[1]][0].split('/')
        for j in labellist:
            row.append(i)
            col.append(basic_f_size + 20 + int(j))
            data.append(1)
	    '''
        #cross gram
        sentence_q = question_info_dic[line[0]][1].split('/')
        sentence_u = user_info_dic[line[1]][1].split('/')
        for m in sentence_q:
            for n in sentence_u:
                t = (m, n)
                sorted(t)
                if cross_dic.has_key(t):
                    row.append(i)
                    col.append(172+cross_dic[t])
                    data.append(1)
	    '''
    return coo_matrix((data, (row, col)), shape=(invited_info_data.shape[0],basic_f_size+20+143))




def create_validate_X(validate_nolabel_data,user_info_dic,question_info_dic):
    basic_f_size = 10
    X_temp = np.zeros((validate_nolabel_data.shape[0],basic_f_size),dtype=np.float32)
    row = []
    col = []
    data = []
    for i,line in enumerate(validate_nolabel_data):
        X_temp[i][0] = float(features_question_dic[line[0]][0])
        X_temp[i][1] = float(features_question_dic[line[0]][1])
        X_temp[i][2] = float(features_expert_dic[line[1]][0])
        X_temp[i][3] = float(features_expert_dic[line[1]][1])
        X_temp[i][4] = return_int_data1(question_info_dic[line[0]][3])
        X_temp[i][5] = return_int_data2(question_info_dic[line[0]][4])
        X_temp[i][6] = return_int_data3(question_info_dic[line[0]][5])
        X_temp[i][7] = return_mean_diff(line[0],line[1])
        X_temp[i][8] = return_max_diff(line[0],line[1])
        X_temp[i][9] = return_lda_sim(line[0],line[1])
       #X_temp[i][10] = return_harmonic_mean(line[0],line[1])
        if i%10000 == 0:
            print i
    X_temp = np.nan_to_num(X_temp)
    min_max_scaler = preprocessing.MinMaxScaler()
    X_temp = min_max_scaler.fit_transform(X_temp)
    for i, line in enumerate(validate_nolabel_data):
        for j in range(basic_f_size):
            row.append(i)
            col.append(j)
            data.append(X_temp[i][j])
        #question  label
        row.append(i)
        col.append(basic_f_size + question_info_dic[line[0]][0])
        data.append(1)
        #user label
        labellist = user_info_dic[line[1]][0].split('/')
        for j in labellist:
            row.append(i)
            col.append(basic_f_size + 20 + int(j))
            data.append(1)
	    '''
        #cross gram
        sentence_q = question_info_dic[line[0]][1].split('/')
        sentence_u = user_info_dic[line[1]][1].split('/')
        for m in sentence_q:
            for n in sentence_u:
                t = (m, n)
                sorted(t)
                if cross_dic.has_key(t):
                    row.append(i)
                    col.append(172+cross_dic[t])
                    data.append(1)
	    '''
    return coo_matrix((data, (row, col)),
                      shape=(validate_nolabel_data.shape[0],basic_f_size+20+143)).toarray()


def create_train_Y(invited_info_data):
    Y = np.empty((invited_info_data.shape[0],),dtype=np.float32)
    for i,line in enumerate(invited_info_data):
        y = line[2]
        Y[i] = y
    return Y



