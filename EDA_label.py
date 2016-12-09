import get_data
import numpy as np


user_info_dic = get_data.get_user_info_dic('./data/user_info.txt')
question_info_dic = get_data.get_question_info_dic('./data/question_info.txt')
invited_info_data = get_data.get_invited_info_data("./data/invited_info_train.txt")
validate_nolabel_data = get_data.get_validata_nolabel_data("./data/validate_nolabel.txt")

list = []
for i,key in enumerate(question_info_dic.keys()):
    list.append(int(question_info_dic[key][0]))
print max(list)
print min(list)

elist = []
for i,key in enumerate(user_info_dic.keys()):
    list = user_info_dic[key][0].split("/")
    for i in range(len(list)):
        #print list[i]
        elist.append(int(list[i]))
print max(elist)
print min(elist)