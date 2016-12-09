import get_data
import numpy as np

user_info_dic = get_data.get_user_info_dic('./data/user_info.txt')
question_info_dic = get_data.get_question_info_dic('./data/question_info.txt')
invited_info_data = get_data.get_invited_info_data("./data/invited_info_train.txt")
validate_nolabel_data = get_data.get_validata_nolabel_data("./data/validate_nolabel.txt")

dic = {}

for i,key in enumerate(user_info_dic.keys()):
    list = user_info_dic[key][1].split('/')
    size = len(list)
    if dic.has_key(size):
        dic[size] +=1
    else:
        dic[size] = 1

for i,key in enumerate(question_info_dic.keys()):
    list = question_info_dic[key][1].split('/')
    size = len(list)
    if dic.has_key(size):
        dic[size] +=1
    else:
        dic[size] = 1

sorted(dic.items(),key=lambda x:x[0])
print dic
