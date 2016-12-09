import get_data

print "get data  ..."
user_info_dic = get_data.get_user_info_dic('./data/user_info.txt')
question_info_dic = get_data.get_question_info_dic('./data/question_info.txt')
invited_info_data = get_data.get_invited_info_data("./data/invited_info_train.txt")


def return_expert_how_good(eid):
    answer = 0.0
    bequestion = 0.0
    for i,line in enumerate(invited_info_data):
        if line[1] == eid:
            bequestion+=1
            if line[2] == 1:
                answer+=1
    try:
        return (answer/bequestion)
    except ZeroDivisionError:
        return 0


def return_expert_label_nums(line):
    label = user_info_dic[line][0]
    labellist = label.split("/")
    return len(labellist)


def return_answer_averagedata(eid):
    sum1 = 0.0
    sum2 = 0.0
    sum3 = 0.0
    count = 1.0
    for i,line in enumerate(invited_info_data):
        if eid == line[1] and line[2] == 1:
            count += 1.0
            sum1 += float(question_info_dic[line[0]][3])
            sum2 += float(question_info_dic[line[0]][4])
            sum3 += float(question_info_dic[line[0]][5])
    print sum1/count,sum2/count,sum3/count
    return sum1/count,sum2/count,sum3/count

def write_f_expert_2_to_txt(f):
    for i,line in enumerate(user_info_dic.keys()):
        print i
        f1 = str(return_expert_how_good(line))
        f2 = str(return_expert_label_nums(line))
        f.write(line+","+f1+","+f2)
        f.write("\n")

def write_f_expert_5_to_txt(f):
    for i,eid in enumerate(user_info_dic.keys()):
        f1 = str(return_expert_how_good(eid))
        f2 = str(return_expert_label_nums(eid))
        return_answer_averagedata(eid)
        #f.write(eid+","+f1+","+f2+","+str(f3)+","+str(f4)+","+str(f5))
        f.write("\n")


#f = open("./features/features_expert.txt",'w')
#write_f_expert_2_to_txt(f)
#f.close()

f = open("./features/features5_expert.txt",'w')
write_f_expert_5_to_txt(f)
f.close()