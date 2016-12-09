import get_data
import numpy as np

invited_info_data = get_data.get_invited_info_data("./data/invited_info_train.txt")


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

invited_info_data = del_negative_to_1_4(invited_info_data)

print invited_info_data.shape[0]