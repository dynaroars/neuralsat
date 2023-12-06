import numpy as np
import torch

class SpecVNNLIB:

    def __init__(self, spec):

        self.bounds, self.mat = spec

        self.prop_mat = None
        self.prop_rhs = None
        self.true_labels = None
        self.target_labels = None


    def extract(self):
        # print('preprocess vnnlib spec')
        assert len(self.mat) == 1
        prop_mat, prop_rhs = self.mat[0]

        if self.prop_mat is None:
            self.prop_mat = torch.tensor(prop_mat, dtype=self.dtype, device=self.device).unsqueeze(0)
        
        if self.prop_rhs is None:
            self.prop_rhs = torch.tensor(prop_rhs, dtype=self.dtype, device=self.device).unsqueeze(0)


        if self.true_labels is None or self.target_labels is None:
            true_labels, target_labels = [], []
            for m in prop_mat:
                true_label = np.where(m == 1)[-1]
                if len(true_label) != 0:
                    assert len(true_label) == 1
                    true_labels.append(true_label[0])
                else:
                    true_labels.append(None)

                target_label = np.where(m == -1)[-1]
                if len(target_label) != 0:
                    assert len(target_label) == 1
                    target_labels.append(target_label[0])
                else:
                    target_labels.append(None)

            self.true_labels = np.array([true_labels])
            self.target_labels = np.array([target_labels])

        # print('c   ', self.prop_mat)
        # print('rhs ', self.prop_rhs)
        # print('y   ', self.true_labels)
        # print('pidx', self.target_labels)
        # exit()

        return self.prop_mat, self.prop_rhs, self.true_labels, self.target_labels


    def check_solution(self, output):
        for lhs, rhs in self.mat:
            lhs = torch.tensor(lhs, dtype=self.dtype, device=output.device)
            rhs = torch.tensor(rhs, dtype=self.dtype, device=output.device)
            vec = lhs @ output.squeeze(0)
            if torch.all(vec <= rhs):
                return True
        return False
