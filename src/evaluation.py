import numpy as np
from sklearn.metrics.cluster import contingency_matrix
from typing import Tuple, Dict, List

class Evaluator:
    """Evaluator class for DeepFashion dataset.
    """

    def __init__(self, metric: str):
        """initialize the evaluator.

        Args:
            metric (str): the name of the metric.
        """
        self.metric = metric

    def __call__(self, 
                 gt_labels: np.ndarray, 
                 pred_labels: np.ndarray):
        """evaluate the clustering performance.

        Args:
            gt_labels (np.ndarray): the ground truth label vector, (N, ).
            pred_labels (np.ndarray): the predicted label vector, (N, ).

        Returns:
            score (float): the clustering performance.
        """
        gt_labels, pred_labels = self.__check(gt_labels, pred_labels)
        
        if self.metric == "pairwise":
            return self.__fowlkes_mallows_score(gt_labels, pred_labels)
        elif self.metric == "bcubed":
            return self.__bcubed(gt_labels, pred_labels)
        else:
            raise ValueError(f"metric {self.metric} is not supported.")
    
    def __check(self, 
                gt_labels: np.ndarray, 
                pred_labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """check the validity of the input.

        Args:
            gt_labels (np.ndarray): the ground truth label vector, (N, ).
            pred_labels (np.ndarray): the predicted label vector, (N, ).
        
        Returns:
            gt_labels (np.ndarray): the ground truth label vector, (N, ).
            pred_labels (np.ndarray): the predicted label vector, (N, ).
        """
        if gt_labels.ndim != 1:
            raise ValueError("gt_labels must be 1D: shape is %r" %
                            (gt_labels.shape, ))
        if pred_labels.ndim != 1:
            raise ValueError("pred_labels must be 1D: shape is %r" %
                            (pred_labels.shape, ))
        if gt_labels.shape != pred_labels.shape:
            raise ValueError(
                "gt_labels and pred_labels must have same size, got %d and %d" %
                (gt_labels.shape[0], pred_labels.shape[0]))

        return gt_labels, pred_labels
    
    def __get_lb2idxs(self, labels: np.ndarray) -> Dict[str, List[int]]:
        """get the label to index mapping. helper function.

        Args:
            labels (np.ndarray): the label vector, (N, ).
        
        Returns:
            lb2idxs (dict): the label to index mapping.
        """
        lb2idxs = {}
        for idx, lb in enumerate(labels):
            if lb not in lb2idxs:
                lb2idxs[lb] = []
            lb2idxs[lb].append(idx)
        return lb2idxs
    
    def __compute_fscore(self, 
                         pre: float, 
                         rec: float) -> float:
        """compute the F-score.

        Args:
            pre (float): the precision.
            rec (float): the recall.
        
        Returns:
            fscore (float): the F-score.
        """
        return 2. * pre * rec / (pre + rec)
    
    def __fowlkes_mallows_score(self, 
                                gt_labels: np.ndarray, 
                                pred_labels: np.ndarray, 
                                sparse: bool = True) -> Tuple[float, float, float]:
        """compute the Fowlkes-Mallows score.

        Args:
            gt_labels (np.ndarray): the ground truth label vector, (N, ).
            pred_labels (np.ndarray): the predicted label vector, (N, ).
            sparse (bool, optional): whether to use sparse matrix. Defaults to True.
        
        Returns:
            avg_pre (float): the average precision.
            avg_rec (float): the average recall.
            fscore (float): the F-score.
        """
        n_samples, = gt_labels.shape

        c = contingency_matrix(gt_labels, pred_labels, sparse=sparse)
        tk = np.dot(c.data, c.data) - n_samples
        pk = np.sum(np.asarray(c.sum(axis=0)).ravel()**2) - n_samples
        qk = np.sum(np.asarray(c.sum(axis=1)).ravel()**2) - n_samples

        avg_pre = tk / pk
        avg_rec = tk / qk
        fscore = self.__compute_fscore(avg_pre, avg_rec)

        return avg_pre, avg_rec, fscore
    
    def __bcubed(self,
                 gt_labels: np.ndarray,
                 pred_labels: np.ndarray) -> Tuple[float, float, float]:
        """compute the B-Cubed score.

        Args:
            gt_labels (np.ndarray): the ground truth label vector, (N, ).
            pred_labels (np.ndarray): the predicted label vector, (N, ).
        
        Returns:
            avg_pre (float): the average precision.
            avg_rec (float): the average recall.
            fscore (float): the F-score.
        """
        gt_lb2idxs = self.__get_lb2idxs(gt_labels)
        pred_lb2idxs = self.__get_lb2idxs(pred_labels)

        num_lbs = len(gt_lb2idxs)
        pre = np.zeros(num_lbs)
        rec = np.zeros(num_lbs)
        gt_num = np.zeros(num_lbs)

        for i, gt_idxs in enumerate(gt_lb2idxs.values()):
            all_pred_lbs = np.unique(pred_labels[gt_idxs])
            gt_num[i] = len(gt_idxs)
            for pred_lb in all_pred_lbs:
                pred_idxs = pred_lb2idxs[pred_lb]
                n = 1. * np.intersect1d(gt_idxs, pred_idxs).size
                pre[i] += n**2 / len(pred_idxs)
                rec[i] += n**2 / gt_num[i]

        gt_num = gt_num.sum()
        avg_pre = pre.sum() / gt_num
        avg_rec = rec.sum() / gt_num
        fscore = self.__compute_fscore(avg_pre, avg_rec)

        return avg_pre, avg_rec, fscore
    


# # Bcubed F-score, precision, recall
# pre, recall, fb = bcubed(gt_labels=label, pred_labels=label)

# # Pairwise F-score, precision, recall
# pre, recall, fp = pairwise(gt_labels=label, pred_labels=label)
