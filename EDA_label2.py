import get_data
import numpy as np


user_info_dic = get_data.get_user_info_dic('./data/user_info.txt')
question_info_dic = get_data.get_question_info_dic('./data/question_info.txt')
invited_info_data = get_data.get_invited_info_data("./data/invited_info_train.txt")
validate_nolabel_data = get_data.get_validata_nolabel_data("./data/validate_nolabel.txt")

#label_beasked_nums = np.zeros((20))
#for j,line in enumerate(invited_info_data):
#    label_beasked_nums[question_info_dic[line[0]][0]]+=1

exp_label_beasked_nums = np.zeros((143))
for i,line in enumerate(invited_info_data):
    labellist = user_info_dic[line[1]][0].split('/')
    for i in range(len(labellist)):
        exp_label_beasked_nums[int(labellist[i])] += 1
print exp_label_beasked_nums