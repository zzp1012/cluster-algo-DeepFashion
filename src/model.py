import numpy as np
from sklearnex import patch_sklearn
patch_sklearn()
import sklearn.cluster as cluster
from sklearn.base import BaseEstimator
from typing import Optional, Union, Tuple, List, Dict

class ModelUtils:
    """Class for model utils. Mainly sklearn models."""

    def __init__(self, model_name: str, **kwargs):
        """Initialize the model.

        Args:
            model_name (str): Name of the model.
        """
        self.model_name = model_name
        self.model = self.auto(model_name, **kwargs)
    
    @classmethod
    def auto(cls,
             model_name: str,
             **kwargs) -> BaseEstimator:
        """Auto load the model. Using scikit-learn Package.
        
        Args:
            model_name (str): Name of the model.
        
        Returns:
            model (nn.Module): The model.
        """
        if model_name == "kmeans":
            return cluster.KMeans(**kwargs)
        elif model_name == "HAC":
            n_clusters = kwargs.get("n_clusters")
            return cluster.AgglomerativeClustering(n_clusters = n_clusters)
        else:
            raise ValueError("model_name not found.")

    @classmethod
    def __check(cls,
                X: np.ndarray,
                y: Optional[np.ndarray] = None):
        """Simple check for the data.

        Args:
            X (np.ndarray): Features. (N, D)
            y (Optional[np.ndarray]): Labels. (N, )
        
        Raises:
            ValueError: If the shape of X and y is not correct.
        """
        if X.ndim != 2:
            raise ValueError(f"X should be 2D. but got {X.ndim}D.")
        
        if y is not None:
            if y.ndim != 1:
                raise ValueError(f"y should be 1D. but got {y.ndim}D.")
            
            if X.shape[0] != y.shape[0]:
                raise ValueError(f"""X and y should have the same length. 
                but got {X.shape[0]} and {y.shape[0]}.""")

    def predict(self, 
                X: np.ndarray) -> np.ndarray:
        """Predict the labels.

        Args:
            X (np.ndarray): Features. (N, D)
        
        Returns:
            y_pred (np.ndarray): Predicted labels. (N, )
        """
        # simple check
        self.__check(X)

        y_pred = self.model.fit_predict(X)
        return y_pred
