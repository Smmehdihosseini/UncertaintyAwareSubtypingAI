import numpy as np

class Uncertainty:

    def __init__(self):

        self.uncertainty = None
        self.metric = None
        self.hyperparams = None
        self.__build_metric__()

    def __build_metric__(self):

        self.metric_dict = {
            "pred_entropy":self.__pred_entropy__,
            "renyi_entropy":self.__renyi__,
            "mutual_info":self.__mi__,
            "tot_var":self.__tv__,
            "mar_conf":self.__moC__
        }

        self.certain_dict = {
            "pred_entropy":0,
            "renyi_entropy":0,
            "mutual_info":np.inf,
            "tot_var":0,
            "mar_conf":0
        }

    def __pstar__(self, predictions):

        return np.mean(predictions, axis=0)
    
    def __pred_entropy__(self, predictions):

        predictions = np.array(predictions)
        return -np.sum(self.__pstar__(predictions) * np.log2(self.__pstar__(predictions) + 1e-9))

    def __renyi__(self, predictions):

        try:
            alpha = self.hyperparams['alpha']
            norm = self.hyperparams['norm']
            num_bins = self.hyperparams['num_bins']
        except KeyError as e:
            raise KeyError("Hyperparameter not found, try to set the hyperparameter first using `.set_hyperparams") from e

        norm_factor = np.log2(num_bins) if norm==True else 1
                
        if alpha <= 0 or alpha == 1:
            raise ValueError("Alpha should be > 0 and != 1")

        predictions = np.array(predictions)
        predictions = np.clip(predictions, 1e-10, 1)
        if alpha == float('inf'):
            return -np.log(np.max(predictions))/norm_factor
        else:
            renyi = (1.0 / (1.0 - alpha)) * np.log(np.sum(predictions ** alpha))
            return renyi/norm_factor
        
    def __mi__(self, predictions):
        predictions = np.array(predictions)
        entropy = -np.sum(self.__pstar__(predictions) * np.log2(self.__pstar__(predictions) + 1e-9))
        expected_entropy = -np.mean(np.sum(predictions * np.log2(predictions + 1e-9), axis=1))
        return entropy - expected_entropy
    
    def __tv__(self, predictions):
        predictions = np.array(predictions)
        return np.mean(np.sum((predictions - self.__pstar__(predictions)) ** 2, axis=1))

    def __moC__(self, predictions):
        predictions = np.array(predictions)
        sorted_preds = np.sort(predictions, axis=1)
        return np.mean(sorted_preds[:, -1] - sorted_preds[:, -2])
    

    def __bhatt_dist__(self, p, q):

        p /= np.sum(p)
        q /= np.sum(q)
        
        BC = np.sum(np.sqrt(p * q))
        BC = min(BC, 1)
        dist = -np.log(BC)
        
        return dist

    def set_metric(self, metric_params):
        """
        Set Metric for Uncertainty Analysis.

        Parameters:
        - metric (str): `pred_entropy`, `renyi_entropy`, `mutual_info`, `tot_var`, `mar_conf`.
        - alpha (float): Applicable to `renyi` entropy. Must be a non-negative number and not equal to 1.
        - norm (bool): Normalization of entropy result.
        - num_bins (bool): Applicable to `renyi` entropy. Num of bins set for histogram to be used for normalization.
        """

        self.metric = self.metric_dict[metric_params['metric']] if 'metric' in metric_params.keys() else None
        self.hyperparams = metric_params
    
    def get_uncertainty(self, predictions):

        if 'threshold' not in self.hyperparams.keys():
            if 'metric' in self.hyperparams.keys():
                return self.certain_dict[self.hyperparams['metric']]
            else:
                return 0
        
        if self.metric==None:
            raise ReferenceError("Uncertainty metric has not been set. Try to set to metric using `.set_metric` method")
        
        self.uncertainty = self.metric(predictions)
        return self.uncertainty
    
    def is_certain(self):
        
        if 'threshold' not in self.hyperparams.keys():
            return True
        
        if self.hyperparams['inequality']=='U<=T':
            if self.uncertainty <= self.hyperparams['threshold']:
                return True
            else:
                return False
        elif self.hyperparams['inequality']=='U>=T':
            if self.uncertainty >= self.hyperparams['threshold']:
                return True
            else:
                return False
        
        return self.uncertainty