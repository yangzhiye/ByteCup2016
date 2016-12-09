from keras.models import Sequential
from keras.layers import Merge,LSTM,Dense,Dropout,Masking,Lambda
from keras.regularizers import l2,activity_l2
import numpy as np
import get_data
import logging
from gensim.models import Word2Vec
import create_172f


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


model = Word2Vec.load("50features_1minwords_10context")
#logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',level=logging.INFO)
user_info_dic = get_data.get_user_info_dic('./data/user_info.txt')
question_info_dic = get_data.get_question_info_dic('./data/question_info.txt')
invited_info_data = get_data.get_invited_info_data("./data/create_train.txt")
validate_nolabel_data = get_data.get_validata_nolabel_data("./data/validate_nolabel.txt")
#create_172f.Init_data(user_info_dic,question_info_dic,invited_info_data,validate_nolabel_data)
print "get train x y and test x..."
train_X = create_172f.create_train_X(invited_info_data, user_info_dic, question_info_dic)
train_Y = create_172f.create_train_Y(invited_info_data)
print train_X[:10]
print train_X.shape
#test_X = create_172f.create_validate_X(validate_nolabel_data, user_info_dic, question_info_dic)

size = invited_info_data.shape[0]


x_train_a = np.zeros((size,30,50))
x_train_b = np.zeros((size,30,50))
y_train = np.zeros((size,2))
data_dim = 50
timesteps = 30
nb_classes = 2



for i,line in enumerate(invited_info_data):
    q_list = question_info_dic[line[0]][1].split('/')
    for j in range(min(timesteps,len(q_list))):
        x_train_a[i][j] = model[q_list[j]]
    e_list = user_info_dic[line[1]][1].split('/')
    for j in range(min(timesteps,len(e_list))):
        x_train_b[i][j] = model[e_list[j]]
    y_train[i][line[2]] = 1


encoder_a = Sequential()
encoder_a.add(Masking(mask_value=0.,input_shape=(timesteps,data_dim)))
encoder_a.add(LSTM(100))
encoder_a.add(Dropout(0.5))

encoder_b = Sequential()
encoder_b.add(Masking(mask_value=0.,input_shape=(timesteps,data_dim)))
encoder_b.add(LSTM(100))
encoder_b.add(Dropout(0.5))

encoder_c = Sequential()
encoder_c.add(Lambda(lambda x:x,input_shape=(172,)))
#encoder_c.add(Dense(200,activation='relu',input_dim=172))
#encoder_c.add(Dropout(0.5))

#decoder = Sequential()
#decoder.add(Merge([encoder_a,encoder_b],mode='concat'))

#decoder.add(Dense(1,activation='sigmoid'))
#decoder.add(Dropout(0.5))

final = Sequential()
final.add(Merge([encoder_a,encoder_b,encoder_c],mode='concat'))
final.add(Dense(2,activation='sigmoid',W_regularizer=l2(0.01),activity_regularizer=activity_l2(0.01)))

final.compile(loss='categorical_crossentropy',
                optimizer='rmsprop',
                metrics=['accuracy'])

final.fit([x_train_a,x_train_b,train_X],y_train,batch_size=1280,nb_epoch=10,validation_split=0.2)
final.save("del_30_0923_6.h5")