import numpy as np
import get_data

user_info_dic = get_data.get_user_info_dic('./data/user_info.txt')
question_info_dic = get_data.get_question_info_dic('./data/question_info.txt')
invited_info_data = get_data.get_invited_info_data("./data/invited_info_train.txt")
validate_nolabel_data = get_data.get_validata_nolabel_data("./data/validate_nolabel.txt")

cross_dic = {}
for i,line in enumerate(invited_info_data):
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
print len(cross_dic)
for k in cross_dic:
    if cross_dic[k] < 100:
        del_list.append(k)

for i in del_list:
    del cross_dic[i]


j = 0
for i in cross_dic:
    cross_dic[i]=j
    j+=1

print len(cross_dic)

