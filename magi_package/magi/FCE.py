import pandas as pd
import numpy as np
import warnings


class FCE:
    def __init__(self, criteria_importance, factor_importance, expert_eval):
        self.RI = (0, 0, 0.58, 0.9, 1.12, 1.24, 1.32, 1.41, 1.45, 1.49)
        self.criteria = criteria_importance
        self.factor = factor_importance
        self.expert_eval = expert_eval
        self.num_criteria = criteria_importance.shape[0]

    def cal_weights(self, input_matrix):
        criteria = np.array(input_matrix)
        n, n1 = criteria.shape
        assert n == n1, '"Criteria Importance Matrix" is not a square matrix'
        for i in range(n):
            for j in range(n):
                if np.abs(criteria[i, j] * criteria[j, i] - 1) > 1e-7:
                    raise ValueError(
                        'The "criterion importance matrix" is not an inverse symmetric matrix, check the position! ({},{})'.format(
                            i, j))

        eigenvalues, eigenvectors = np.linalg.eig(criteria)

        max_idx = np.argmax(eigenvalues)
        max_eigen = eigenvalues[max_idx].real
        eigen = eigenvectors[:, max_idx].real
        eigen = eigen / eigen.sum()

        if n > 9:
            CR = None
            warnings.warn('Unable to accurately judge consistency')
        else:
            CI = (max_eigen - n) / (n - 1)
            CR = CI / self.RI[n - 1]
        return max_eigen, CR, eigen

    def cal_Uiweights(self, input_matrix):
        max_eigen, CR, eigen = self.cal_weights(input_matrix)
        return max_eigen, CR, eigen

    def cal_Uijweights(self, factor_importance):
        max_eigen = []
        CR = []
        eigen = []
        for i in range(5):
            imax_eigen, iCR, ieigen = self.cal_Uiweights(factor_importance[i])
            max_eigen.append(imax_eigen)
            CR.append(iCR)
            eigen.append(ieigen)
        return max_eigen, CR, eigen

    def fuzzy_eval(self, criteria_weights, factor_weights, expert_eval):
        # score = [1, 0.8, 0.6, 0.4, 0.2]
        expert_eval_df = pd.read_csv(expert_eval)
        print('Single-factor fuzzy comprehensive evaluation:{}\n'.format(expert_eval_df))
        v1 = expert_eval_df.iloc[0:5, :].values
        v2 = expert_eval_df.iloc[5:8, :].values
        v3 = expert_eval_df.iloc[8:10, :].values
        v4 = expert_eval_df.iloc[10:13, :].values
        v5 = expert_eval_df.iloc[13:16, :].values

        vv = [v1, v2, v3, v4, v5]

        val = []
        num = len(factor_weights)
        for i in range(num):
            v = np.dot(np.array(factor_weights[i]), vv[i])
            print('The criterion {} , the matrix product is: {}'.format(i + 1, v))
            val.append(v)

        # target level
        obj = np.dot(criteria_weights, np.array(val))
        print('Fuzzy comprehensive evaluation at the target level: {}\n'.format(obj))
        # comprehensive evaluation
        # fce_eval = np.dot(np.array(obj), np.array(score).T)
        # print('comprehensive evaluation: {}'.format(fce_eval * 100))
        # return fce_eval
        return obj

    def run(self):
        cmax_eigen, CR_c, criteria_eigen = self.cal_Uiweights(self.criteria)
        print('=' * 10 + 'Guideline layer' + '=' * 10)
        print('Maximum eigenvalue: {:<5f}, CR={:<5f}, check {} passed'.format(cmax_eigen, CR_c,
                                                                              '' if CR_c < 0.1 else 'not'))
        print('criterion layer weights = {}\n'.format(criteria_eigen))

        fmax_eigen, CR_f, factor_eigen = self.cal_Uijweights(self.factor)
        for i in range(5):
            print('=' * 10 + 'Guideline layer' + '=' * 10)
            print('Maximum eigenvalue: {:<5f}, CR={:<5f}, check {} passed'.format(fmax_eigen[i], CR_f[i],
                                                                                  '' if CR_f[i] < 0.1 else 'not'))
            print('criterion layer weights = {}\n'.format(factor_eigen[i]))

        fec_eval = self.fuzzy_eval(criteria_eigen, factor_eigen, self.expert_eval)
        print(f'fce done: {fec_eval}')
        return fec_eval
