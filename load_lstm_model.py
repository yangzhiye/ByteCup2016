from keras.models import load_model
import get_data
import numpy as np
from gensim.models import Word2Vec


model = Word2Vec.load("50features_1minwords_10context")
user_info_dic = get_data.get_user_info_dic('./data/user_info.txt')
question_info_dic = get_data.get_question_info_dic('./data/question_info.txt')
invited_info_data = get_data.get_invited_info_data("./data/invited_info_train.txt")
validate_nolabel_data = get_data.get_validata_nolabel_data("./data/validate_nolabel.txt")

Xa = np.zeros((30466,20,50),dtype=np.float32)
Xb = np.zeros((30466,20,50),dtype=np.float32)
Y = np.zeros((30466,2),dtype=np.float32)
timesteps = 20
lstmmodel = load_model("del_20_1280_5_0.1.h5")
for i,line in enumerate(validate_nolabel_data):
    q_list = question_info_dic[line[0]][1].split('/')
    for j in range(min(timesteps,len(q_list))):
        Xa[i][j] = model[q_list[j]]
    e_list = user_info_dic[line[1]][1].split('/')
    for j in range(min(timesteps,len(e_list))):
        Xb[i][j] = model[e_list[j]]
Y = lstmmodel.predict_proba([Xa,Xb])

print Y.shape
print Y[:10]