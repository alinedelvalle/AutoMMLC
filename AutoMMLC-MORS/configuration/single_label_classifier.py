import numpy as np

from sklearn.tree import DecisionTreeClassifier
   

def get_config_dict(n_features, n_labels):
    sl_classifier_config_dict = {
    
        # Auto sklearn
        
        'sklearn.ensemble.RandomForestClassifier': {
            'criterion': ['gini', 'entropy'],
            'max_features': np.around(np.arange(0.1, 1.01, 0.1), 1), 
            'min_samples_split': list(range(2, 21)),
            'min_samples_leaf': list(range(1, 21)),
            'bootstrap': [True, False]
        },
        
        
        'sklearn.neural_network.MLPClassifier':{
            'activation': ['tanh', 'relu', 'logistic'],
            'alpha': [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
            'early_stopping': [True, False],
            'learning_rate_init': [1e-4, 1e-3, 1e-2, 1e-1, 0.2, 0.3, 0.4, 0.5],
            'hidden_layer_sizes': [n_labels, int((n_labels+n_features)/2), n_features, n_features+n_labels],  
            'max_iter': [3000]
        },


        'sklearn.linear_model.SGDClassifier': {
            'alpha': [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
            'average': [True, False],
            'epsilon': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
            'eta0': [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
            'l1_ratio': [1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1],
            'learning_rate': ['optimal', 'invscaling', 'constant'],
            'loss': ['log'],
            'penalty': ['l2', 'l1', 'elasticnet'],
            'power_t': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1],
            'tol': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
        },


        'sklearn.ensemble.ExtraTreesClassifier':{
            'criterion': ['gini', 'entropy'],
            'max_features': np.around(np.arange(0.05, 1.01, 0.05), 2),
            'min_samples_split': list(range(2, 21)),
            'min_samples_leaf': list(range(1, 21)),
            'bootstrap': [True, False]
        },

        # TPOT    
        'sklearn.neighbors.KNeighborsClassifier': {
            'n_neighbors': list(range(1, 101, 2)),
            'weights': ['uniform', 'distance'],
            'p': [1, 2, 3]
        },


        'sklearn.tree.DecisionTreeClassifier': {
            'criterion': ['gini', 'entropy'],
            'max_depth': range(1, 11),
            'min_samples_split': list(range(2, 21)),
            'min_samples_leaf': list(range(1, 21))
        }, 


        'sklearn.ensemble.AdaBoostClassifier': {
            'learning_rate': [1e-3, 1e-2, 1e-1, 0.5, 1.0],
            'algorithm': ['SAMME.R', 'SAMME'],
            'base_estimator': [DecisionTreeClassifier(max_depth= n) for n in range(1,11)]
        },


        'sklearn.naive_bayes.BernoulliNB':{
            'alpha': [1e-3, 1e-2, 1e-1, 1, 10, 100],
            'fit_prior': [True, False]
        },


        'sklearn.naive_bayes.MultinomialNB': {
            'alpha': [1e-3, 1e-2, 1e-1, 1, 10, 100],
            'fit_prior': [True, False]
        },

        # Bogatinovski (2022)
        'sklearn.svm.SVC': {
            'kernel': ['rbf'],
            'C' : [2e-5, 2e-3, 2e-1, 2e1, 2e3, 2e5, 2e7, 2e9, 2e11, 2e13, 2e15], 
            'gamma' : [2e-15, 2e-13, 2e-11, 2e-9, 2e-7, 2e-5, 2e-3, 2e-1, 2e1, 2e3]
        },

        # Gama
        'sklearn.linear_model.LogisticRegression': {
            'penalty': ['l2'],
            'C': [1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1.0, 5.0, 10.0, 15.0, 20.0, 25.0],
            'dual': [False],
            'solver': ['newton-cg', 'lbfgs', 'sag', 'saga'],
            'max_iter': [3000]
        }
        
    }
        
    return sl_classifier_config_dict


