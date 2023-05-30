import os
import numpy as np
from typing import Tuple, List

# import internal libs
from utils import get_logger

class DataUtils:
    """Data utils class for DeepFashion dataset.
    """
    
    @classmethod
    def read(cls,
             feat_path: str = "../data/feat.bin", 
             label_path: str = "../data/label.meta"):
        """read the raw data from the data folder.

        Args:
            feat_path (str): the path of the feature file.
            label_path (str): the path of the label file.
        
        Returns:
            feat (np.ndarray): the feature matrix, (N, D).
            label (np.ndarray): the label vector, (N, ).
        """
        # setup the logger
        logger = get_logger(f"{__name__}.{cls.__name__}.read")

        label = cls.__read_meta(label_path)
        inst_num = len(label)
        feat = cls.__read_probs(feat_path, inst_num=inst_num, feat_dim=256)

        logger.info(f"feature shape: {feat.shape}; label number: {len(np.unique(label))}")
        return feat, label

    @classmethod
    def __read_probs(cls,
                     path: str,
                     inst_num: int,
                     feat_dim: int,
                     dtype=np.float32,
                     verbose=False):
        """read the feature file.

        Args:
            path (str): the path of the feature file.
            inst_num (int): the number of instances.
            feat_dim (int): the dimension of the feature.
            dtype (np.dtype): the data type of the feature.
            verbose (bool): whether to print the information of the feature.
        
        Returns:
            probs (np.ndarray): the feature matrix, (N, D).
        """
        assert (inst_num > 0 or inst_num == -1) and feat_dim > 0, \
            "inst_num and feat_dim must be greater than 0."
        
        # setup the logger
        logger = get_logger(f"{__name__}.{cls.__name__}.__read_probs", verbose=verbose)
        
        count = -1
        if inst_num > 0:
            count = inst_num * feat_dim
            logger.info(f"count: {count}")
        probs = np.fromfile(path, dtype=dtype, count=count)
        if feat_dim > 1:
            probs = probs.reshape(inst_num, feat_dim)
        logger.debug(f"[{path}] shape: {probs.shape}")
        return probs
    
    @classmethod
    def __read_meta(cls,
                    fn_meta: str,
                    start_pos: int = 0,
                    verbose: bool = True):
        """read the label file.

        Args:
            fn_meta (str): the path of the label file.
            start_pos (int): the start position of the label file.
            verbose (bool): whether to print the information of the label.
        
        Returns:
            labels (np.ndarray): the label vector, (N, ).
        """
        # setup the logger
        logger = get_logger(f"{__name__}.{cls.__name__}.__read_meta", verbose=verbose)

        lb2idxs = {}
        idx2lb = {}
        with open(fn_meta) as f:
            for idx, x in enumerate(f.readlines()[start_pos:]):
                lb = int(x.strip())
                if lb not in lb2idxs:
                    lb2idxs[lb] = []
                lb2idxs[lb] += [idx]
                idx2lb[idx] = lb
        
        inst_num = len(idx2lb)
        cls_num = len(lb2idxs)
        labels = np.zeros(inst_num)
        for i, label in idx2lb.items():
            labels[i] = label
        
        logger.debug(f"[{fn_meta}] #cls: {cls_num}, #inst: {inst_num}")
        return labels


if __name__ == "__main__":
    # read the data
    feat, label = DataUtils.read()