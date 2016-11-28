from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm
import create_features
import get_data



print "get data  ..."
user_info_dic = get_data.get_user_info_dic('./data/user_info.txt')
question_info_dic = get_data.get_question_info_dic('./data/question_info.txt')
invited_info_data = get_data.get_invited_info_data("./data/invited_info_train.txt")
validate_nolabel_data = get_data.get_validata_nolabel_data("./data/final_data.txt")

print 'del negivate ...'
#invited_info_data = create_8f_deln_0830.del_negative_to_1_4(invited_info_data)

print "get train x y and test x..."
#train_X = create_173f.create_train_X(invited_info_data, user_info_dic, question_info_dic)
train_Y = create_features.create_train_Y(invited_info_data)
#print train_X[:10]
#test_X = create_173f.create_validate_X(validate_nolabel_data, user_info_dic, question_info_dic)


print "fit train x y..."
#0.49328
clf = LogisticRegression()
#clf = svm.SVC(probability=True)
#clf = GradientBoostingClassifier(n_estimators = 300 , max_depth = 50)
clf.fit(create_features.create_train_X(invited_info_data, user_info_dic, question_info_dic), train_Y)

print "predict_proba test x..."
result = clf.predict_proba(create_features.create_validate_X(validate_nolabel_data, user_info_dic, question_info_dic))

print result

f = open("./result/final_1115.txt",'w')

for i,line in enumerate(result):
    f.write(str(line[1]))
    f.write('\n')
f.close()

