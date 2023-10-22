import numpy as np
from sklearn.metrics import confusion_matrix

class _StreamMetrics(object):
    def __init__(self):
        """ Overridden by subclasses """
        raise NotImplementedError()

    def update(self, gt, pred):
        """ Overridden by subclasses """
        raise NotImplementedError()

    def get_results(self):
        """ Overridden by subclasses """
        raise NotImplementedError()

    def to_str(self, metrics):
        """ Overridden by subclasses """
        raise NotImplementedError()

    def reset(self):
        """ Overridden by subclasses """
        raise NotImplementedError()      

class StreamSegMetrics(_StreamMetrics):
    """
    Stream Metrics for Semantic Segmentation Task
    """
    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.confusion_matrix = np.zeros((n_classes, n_classes))

    def update(self, label_trues, label_preds):
        for lt, lp in zip(label_trues, label_preds):
            self.confusion_matrix += self._fast_hist( lt.flatten(), lp.flatten() )
    
    @staticmethod
    def to_str(results):
        string = "\n"
        for k, v in results.items():
            if k!="Class IoU":
                string += "%s: %f\n"%(k, v)

        return string

    def _fast_hist(self, label_true, label_pred):
        mask = (label_true >= 0) & (label_true < self.n_classes)

        # row is target, col is pred
        hist = np.bincount(
            self.n_classes * label_true[mask].astype(int) + label_pred[mask],
            minlength=self.n_classes ** 2,
        ).reshape(self.n_classes, self.n_classes)
        return hist

    def get_results(self):
        """Returns accuracy score evaluation result.
            - overall accuracy
            - mean accuracy
            - mean IU
            - fwavacc
        """
        hist = self.confusion_matrix

        print('confusion_matrix==================================================')
        print(hist)
        print('==================================================================')
        print('Row sum is target, and colum sum is predict. ')

        hot = 1e-6

        acc = np.diag(hist).sum() / hist.sum()
        acc_cls = (np.diag(hist)) / (hist.sum(axis=1))
        iu = (np.diag(hist)) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
        mean_iu = np.nanmean(iu)


        TN, FP, FN, TP = hist[0, 0], hist[0, 1], hist[1, 0], hist[1, 1]
        precision = TP/(TP+FP+hot)
        recall = TP/(TP+FN+hot)
        F_score = (2 * precision * recall) / (precision + recall+hot)


        def Kappa(matrix):
            all_sum = np.sum(matrix)
            sum_po = np.diag(matrix).sum()
            sum_pe = np.sum(matrix.sum(axis=0) * matrix.sum(axis=1))
            po = (sum_po) / (all_sum)
            pe = (sum_pe) / (all_sum * all_sum)
            return (po - pe) / (1 - pe)

        # def producerAcc(matrix):
        #     diag_value = np.diag(matrix)
        #     col_sum = matrix.sum(axis=0)
        #     p_acc = (diag_value) / (col_sum)
        #     p_acc = np.nanmean(p_acc)
        #     return p_acc
        #
        # def userAcc(matrix):
        #     diag_value = np.diag(matrix)
        #     row_sum = matrix.sum(axis=1)
        #     user_acc = (diag_value) / (row_sum)
        #     user_acc = np.nanmean(user_acc)
        #     return user_acc


        # # producer_acc same as Recall,
        # # user_acc same as prcision
        # producer_acc = producerAcc(hist)
        # user_acc = userAcc(hist)
        kappa_coefficient = Kappa(hist)


        return {'OA': acc,
                'mIoU': mean_iu,
                'F1': F_score,
                'Ka': kappa_coefficient,
                'Pr':precision,
                'Re': recall,
                }
    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))

class AverageMeter(object):
    """Computes average values"""
    def __init__(self):
        self.book = dict()

    def reset_all(self):
        self.book.clear()
    
    def reset(self, id):
        item = self.book.get(id, None)
        if item is not None:
            item[0] = 0
            item[1] = 0

    def update(self, id, val):
        record = self.book.get(id, None)
        if record is None:
            self.book[id] = [val, 1]
        else:
            record[0]+=val
            record[1]+=1

    def get_results(self, id):
        record = self.book.get(id, None)
        assert record is not None
        return record[0] / record[1]
