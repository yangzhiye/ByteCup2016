import get_data
import numpy as np
from sklearn import svm

user_info_dic = get_data.get_user_info_dic('./data/user_info.txt')
question_info_dic = get_data.get_question_info_dic('./data/question_info.txt')
invited_info_data = get_data.get_invited_info_data("./data/invited_info_train.txt")
validate_nolabel_data = get_data.get_validata_nolabel_data("./data/validate_nolabel.txt")

def cul_positive_negative(invited_info_data):
    positive = 0
    negative = 0
    for i,line in enumerate(invited_info_data):
        #print invited_info_data[i][2]
        if invited_info_data[i][2] == 0:
            negative+=1
        else:
            positive+=1
    print "positive  :  ",positive,"negative  :  ",negative

def del_negative_100000(invited_info_data):
    # positive 27324   nagative  218428 to 118428
    del_num = 0
    data = invited_info_data
    for i,line in enumerate(data):
        if del_num==100000:
            break
        if invited_info_data[i][2] == 0:
            invited_info_data = np.delete(invited_info_data,i,0)
            del_num+=1
            print del_num
        else:
            pass
    return invited_info_data

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

cul_positive_negative(invited_info_data)
invited_info_data = del_negative_to_1_4(invited_info_data)
cul_positive_negative(invited_info_data)
print invited_info_data[:100]
print invited_info_data.shape