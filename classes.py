from collections import Counter
import numpy as np


class Repository:
    def __init__(self):
        self.vuln_lst = []
        self.benign_lst = []
        self.vuln_details = []
        self.benign_details = []
        self.file = ""

    def pad_repo(self):
        padded_vuln_all, padded_benign_all = [], []
        to_pad = max(max(Counter([v.shape[0] for v in self.vuln_lst])),
                     max(Counter([v.shape[0] for v in self.benign_lst])))

        for vuln in self.vuln_lst:
            padded_vuln_all.append(np.pad(vuln, ((to_pad - vuln.shape[0], 0), (0, 0))))

        for benign in self.benign_lst:
            padded_benign_all.append(np.pad(benign, ((to_pad - benign.shape[0], 0), (0, 0))))

        self.vuln_lst = np.nan_to_num(np.array(padded_vuln_all))
        self.benign_lst = np.nan_to_num(np.array(padded_benign_all))

    def get_all_lst(self):
        X = np.concatenate([self.vuln_lst,self.benign_lst])
        y = len(self.vuln_lst) * [1] + len(self.benign_lst)*[0]
        return X, y
