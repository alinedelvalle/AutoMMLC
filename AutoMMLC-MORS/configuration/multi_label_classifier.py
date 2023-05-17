import numpy as np

import configuration.single_label_classifier as config_single


def get_single_classifier(n_features, n_labels):
    return np.array(list(config_single.get_config_dict(n_features, n_labels).keys()))
    

def get_config_dict(n_features, n_labels):
    ml_classifier_config_dict = {
        
        # Adaptation Problem
        'skmultilearn.adapt.BRkNNaClassifier': {
            'k': list(range(1, 31, 1))
        },


        'skmultilearn.adapt.BRkNNbClassifier': {
            'k': list(range(1, 31, 1))
        },


        'skmultilearn.adapt.MLkNN': {
            'k': list(range(6, 21, 2)),
        },

        'skmultilearn.adapt.MLARAM': {
            'vigilance': [0.001, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95],
            'threshold': [0.001, 0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
        },

        'skmultilearn.adapt.MLTSVM': {
            'c_k': [2e-6, 2e-5, 2e-4, 2e-3, 2e-2, 2e-2, 1, 2e1, 2e2, 2e3, 2e4, 2e5, 2e6], # penalidade
            'sor_omega': [0.2], #np.around(np.arange(0, 2.01, 0.1),1),
            'lambda_param': [2e-4, 2e-3, 2e-2, 2e-2, 1, 2e1, 2e2, 2e3, 2e4] # regularização
        }, 

        # Transformation Problem

        'skmultilearn.problem_transform.BinaryRelevance': {
            'classifier': get_single_classifier(n_features, n_labels)
        },


        'skmultilearn.problem_transformation.ClassifierChain': {
            'classifier' : get_single_classifier(n_features, n_labels),
        },


        # Ensemble

        'skmultilearn.ensemble.RakelD': {
            'base_classifier': get_single_classifier(n_features, n_labels),
            'labelset_size': [3] # list(range(int(np.ceil(n_labels/10)), int(np.ceil(0.9*n_labels))))
        }, 


        # 'skmultilearn.ensemble.RakelO': {
        #    'base_classifier': get_single_classifier(n_features, n_labels),
        #    'labelset_size': [3], 
        #    'model_count': list(range(n_labels, 2*n_labels))
        #},


        # 'skmultilearn.problem_transformation.LabelPowerset': {
        #    'classifier': get_single_classifier(n_features, n_labels),
        # }
        
        'sklearn.ensemble.RandomForestClassifier': {
            'criterion': ['gini', 'entropy'],
            'max_features': np.around(np.arange(0.1, 1.01, 0.1), 1), 
            'min_samples_split': list(range(2, 21)),
            'min_samples_leaf': list(range(1, 21)),
            'bootstrap': [True, False]
        },
        
        'sklearn.tree.DecisionTreeClassifier': {
            'criterion': ['gini', 'entropy'],
            'max_depth': range(1, 11),
            'min_samples_split': list(range(2, 21)),
            'min_samples_leaf': list(range(1, 21))
        }
           
    }
    
    return ml_classifier_config_dict

