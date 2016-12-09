import get_data

print "get data  ..."
user_info_dic = get_data.get_user_info_dic('./data/user_info.txt')
question_info_dic = get_data.get_question_info_dic('./data/question_info.txt')
invited_info_data = get_data.get_invited_info_data("./data/invited_info_train.txt")


def return_question_how_good(qid):
    beanswer = 0.0
    bequestion = 0.0
    for i,line in enumerate(invited_info_data):
        if line[0] == qid:
            bequestion+=1
            if line[2] == 1:
                beanswer+=1
    try:
        return (beanswer/bequestion)
    except ZeroDivisionError:
        return 0
def return_question_label_how_good(label):
    all = 0.0
    beanswer = 0.0
    for i,line in enumerate(invited_info_data):
        if question_info_dic[line[0]][0] == label:
            all+=1
            if line[2] == 1:
                beanswer+=1
    try:
        return (beanswer/all)
    except ZeroDivisionError:
        return 0



def write_f_question_2_to_txt(f):
    for i,line in enumerate(question_info_dic.keys()):
        print i
        f1 = str(return_question_how_good(line))
        f2 = str(return_question_label_how_good(question_info_dic[line][0]))
        f.write(line+","+f1+","+f2)
        f.write("\n")


f = open("./features/features_question.txt",'w')
write_f_question_2_to_txt(f)
f.close()