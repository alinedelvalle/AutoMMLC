import numpy as np

import configuration.single_label_classifier as config_single

import configuration.multi_label_classifier as config_multi


class Configuration:
    
    def __init__(self, n_f, n_l):
        self.n_features = n_f # feature
        self.n_labels = n_l # label
    
    def get_sl_algorithms(self):
        return np.array(list(config_single.get_config_dict(self.n_features, self.n_labels).keys()))
    
    def get_ml_algorithms(self):
        return np.array(list(config_multi.get_config_dict(self.n_features, self.n_labels).keys()))
    
    def get_all_algorithms(self):
        return np.concatenate([self.get_ml_algorithms(), self.get_sl_algorithms()], axis=0)
    
    # dicionários
    def get_sl_classifier_config(self):
        return config_single.get_config_dict(self.n_features, self.n_labels)
    
    def get_ml_classifier_config(self):
        return config_multi.get_config_dict(self.n_features, self.n_labels) 