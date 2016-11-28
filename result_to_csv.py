import numpy as np
from sklearn import preprocessing
import get_data

def result_to_normalization(filepath):
    f = open(filepath)
    data = np.empty((30167),dtype=np.float32)
    for i,line in enumerate(f.readlines()):
        data[i] = float(line)
    min_max_scaler = preprocessing.MinMaxScaler()
    data = min_max_scaler.fit_transform(data)
    f.close()
    return data


def get_result_from_txt(filepath):
    f = open(filepath)
    data = np.empty((30167),dtype=np.float32)
    for i,line in enumerate(f.readlines()):
        data[i] = float(line)
    f.close()
    return data


def result_to_csv(f,data):
    validate_nolabel_data = get_data.get_validata_nolabel_data("./data/final_data.txt")
    for i,line in enumerate(data):
        f.write(validate_nolabel_data[i][0]+","+validate_nolabel_data[i][1]+","+str(data[i]))
        f.write("\n")

#no norm
#f = open("./csv_result/temp_4f_svm_no-norm_0827.csv", 'w')
#f = open("./csv_result/temp_4f_svm_0827.csv",'w')
#f = open("./csv_result/temp_16f_svm_0828.csv",'w')
f = open("./csv_result/final_1115.csv",'w')
f.write("qid,uid,label")
f.write("\n")
#data = get_result_from_txt("./result/4f_svm_result_0827.txt")
#data = result_to_normalization("./result/4f_svm_result_0827.txt")
#data = get_result_from_txt("./result/16f_svm_result_0828.txt")
data = get_result_from_txt("./result/final_1115.txt")
result_to_csv(f,data)
f.close()
