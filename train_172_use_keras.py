from sklearn.linear_model import LogisticRegression
from keras.models import Sequential
from keras.layers import Merge,LSTM,Dense,Dropout,Masking,Lambda
from keras.regularizers import l2,activity_l2
import create_172f
import get_data

print "get data  ..."
user_info_dic = get_data.get_user_info_dic('./data/user_info.txt')
question_info_dic = get_data.get_question_info_dic('./data/question_info.txt')
invited_info_data = get_data.get_invited_info_data("./data/create_train.txt")
validate_nolabel_data = get_data.get_validata_nolabel_data("./data/validate_nolabel.txt")

print 'del negivate ...'
#invited_info_data = create_8f_deln_0830.del_negative_to_1_4(invited_info_data)

print "get train x y and test x..."
train_X = create_172f.create_train_X(invited_info_data, user_info_dic, question_info_dic)
train_Y = create_172f.create_train_Y(invited_info_data)
print train_X[:10]
test_X = create_172f.create_validate_X(validate_nolabel_data, user_info_dic, question_info_dic)

model = Sequential()
model.add(Dense(100,init='uniform',activation='relu',input_dim=172))
model.add(Dropout(0.5))
model.add(Dense(100,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1,activation='sigmoid'))

model.compile(loss='binary_crossentropy',optimizer='rmsprop',metrics=['accuracy'])
model.save("Dense_model_1.h5")

#print "fit train x y..."
#clf = LogisticRegression()
#clf.fit(train_X,train_Y)

#print "predict_proba test x..."
#result = clf.predict_proba(test_X)

#print result

#f = open("./result/173f_lr_result_0920_2.txt",'w')
#f = open("./result/20f_lr_result_0829.txt",'w')
#for i,line in enumerate(result):
#    f.write(str(line[1]))
#    f.write('\n')
#f.close()

