# adapted from https://github.com/yiyanxu/diffrec
import numpy as np
import scipy.sparse as sp
from torch.utils.data import Dataset

def data_load(train_path, valid_path, test_path):
    train_list = np.load(train_path, allow_pickle=True)
    valid_list = np.load(valid_path, allow_pickle=True)
    test_list = np.load(test_path, allow_pickle=True)

    uid_max = 0
    iid_max = 0
    
    for uid, iid in train_list:
        if uid > uid_max:
            uid_max = uid
        if iid > iid_max:
            iid_max = iid
    
    # Check validation data
    for uid, iid in valid_list:
        if uid > uid_max:
            uid_max = uid
        if iid > iid_max:
            iid_max = iid
    
    for uid, iid in test_list:
        if uid > uid_max:
            uid_max = uid
        if iid > iid_max:
            iid_max = iid
    
    n_user = uid_max + 1
    n_item = iid_max + 1
    
    train_dict = {}
    for uid, iid in train_list:
        if uid not in train_dict:
            train_dict[uid] = []
        train_dict[uid].append(iid)
    print(f'user num: {n_user}')
    print(f'item num: {n_item}')

    train_data = sp.csr_matrix((np.ones_like(train_list[:, 0]),
        (train_list[:, 0], train_list[:, 1])), dtype='float64',
        shape=(n_user, n_item))
    
    valid_y_data = sp.csr_matrix((np.ones_like(valid_list[:, 0]),
                 (valid_list[:, 0], valid_list[:, 1])), dtype='float64',
                 shape=(n_user, n_item))  # valid_ground truth

    test_y_data = sp.csr_matrix((np.ones_like(test_list[:, 0]),
                 (test_list[:, 0], test_list[:, 1])), dtype='float64',
                 shape=(n_user, n_item))  # test_ground truth
    
    return train_data, valid_y_data, test_y_data, n_user, n_item


class DataDiffusion(Dataset):
    def __init__(self, data):
        self.data = data
    def __getitem__(self, index):
        item = self.data[index]
        return item
    def __len__(self):
        return len(self.data)
