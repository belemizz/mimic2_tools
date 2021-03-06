'''Class definition for evaluation'''
import numpy as np
from mutil import float_div
from sklearn import metrics


class BinaryClassResult():
    '''Manages classification result'''

    def __init__(self, ground_truth, prediction=None, score=None):
        '''initializer

        :param ground_truth: ground ground_truth label
        :param prediction: prediction by algorithm
        :param score: prediction score by algorithm
        '''

        self.gt = np.array(ground_truth)

        if prediction is not None:
            self.pred = np.array(prediction)
            self.calc_metrics()

        if score is not None:
            self.score = np.array(score)
            self.__calc_roc()

    def calc_metrics(self):
        self.n_posi = int(sum(self.gt == 1))
        self.n_nega = int(sum(self.gt == 0))
        self.n_tp = int(sum(self.pred[self.gt == 1]))
        self.n_fp = int(sum(self.pred[self.gt == 0]))
        self.n_tn = self.n_nega - self.n_fp
        self.n_fn = self.n_posi - self.n_tp

        self.recall = float_div(self.n_tp, self.n_posi)
        self.prec = float_div(self.n_tp, (self.n_tp + self.n_fp))
        self.f = float_div(2 * self.prec * self.recall, self.prec + self.recall)
        self.acc = float_div(self.n_tp + self.n_tn, self.n_posi + self.n_nega)

    def __calc_roc(self):
        self.auc = metrics.roc_auc_score(self.gt, self.score)

    def get_dict(self):
        result_dict = {}
        try:
            result_dict.update({'recall': self.recall, 'prec': self.prec,
                                'f': self.f, 'acc': self.acc})
        except:
            pass

        try:
            result_dict.update({'auc': self.auc})
        except:
            pass

        return result_dict


class BinaryClassCVResult(BinaryClassResult):
    '''Manages cross varidation result'''

    def __init__(self, l_binary_class_result):
        self.gt = np.hstack([result.gt for result in l_binary_class_result])

        try:
            self.pred = np.hstack([result.pred for result in l_binary_class_result])
            self.calc_metrics()
        except AttributeError:
            pass

        try:
            self.l_auc = [result.auc for result in l_binary_class_result]
            self.auc = np.mean(self.l_auc)
            self.l_roc_input = [[result.gt, result.score] for result in l_binary_class_result]

            # do I really need this ?
            self.__calc_roc()
        except AttributeError:
            pass

    def __calc_roc(self):
        mean_tpr = 0.0
        mean_fpr = np.linspace(0, 1, 10000)

        for roc_input in self.l_roc_input:
            fpr, tpr, thresholds = metrics.roc_curve(roc_input[0], roc_input[1])
            mean_tpr += np.interp(mean_fpr, fpr, tpr)
            mean_tpr[0] = 0.0

        mean_tpr /= len(self.l_roc_input)
        mean_tpr[-1] = 1.0
        self.mean_auc2 = metrics.auc(mean_fpr, mean_tpr)
