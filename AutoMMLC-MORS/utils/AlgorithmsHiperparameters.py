import numpy as np

import sklearn.metrics as metrics


class AlgorithmsHiperparameters:
    
    list_metrics = []
    #id_general = 1
    
    def add_metrics(id_general, prediction, y_test, time_execution, string_classifier):
        # example
        hmm_loss = metrics.hamming_loss(y_test, prediction)
        accuracy = metrics.accuracy_score(y_test, prediction)
        jaccard_mi = metrics.jaccard_score(y_test, prediction, average='micro')
        precision = metrics.precision_score(y_test, prediction, average='samples', zero_division=0)
        recall = metrics.recall_score(y_test, prediction, average='samples')
        f_score = metrics.f1_score(y_test, prediction, average='samples')
              
        # label
        precision_mi = metrics.precision_score(y_test, prediction, average='micro', zero_division=0)
        recall_mi = metrics.recall_score(y_test, prediction, average='micro')
        f_score_mi = metrics.f1_score(y_test, prediction, average='micro')
        precision_ma = metrics.precision_score(y_test, prediction, average='macro', zero_division=0)
        recall_ma = metrics.recall_score(y_test, prediction, average='macro')
        f_score_ma = metrics.f1_score(y_test, prediction, average='macro')
            
        # rank            
        if isinstance(prediction, np.ndarray):
            rank_loss = metrics.label_ranking_loss(y_test.toarray(), prediction)
            coverage = metrics.coverage_error(y_test.toarray(), prediction)
        else:
            rank_loss = metrics.label_ranking_loss(y_test.toarray(), prediction.toarray())
            coverage = metrics.coverage_error(y_test.toarray(), prediction.toarray())
            
        # str_values = str(AlgorithmsHiperparameters.id_general)+'\t'+str(hmm_loss)+'\t'
        str_values = str(id_general)+'\t'+str(hmm_loss)+'\t'
        str_values += str(accuracy)+'\t'+str(jaccard_mi)+'\t'+str(precision)+'\t'
        str_values += str(recall)+'\t'+str(f_score)+'\t'+str(precision_mi)+'\t'+str(recall_mi)+'\t'
        str_values += str(f_score_mi)+'\t'+str(precision_ma)+'\t'+str(recall_ma)+'\t'+str(f_score_ma)+'\t'
        str_values += str(rank_loss)+'\t'+str(coverage)+'\t'+str(time_execution)+'\t'+string_classifier+'\n'
            
        AlgorithmsHiperparameters.list_metrics = np.append(AlgorithmsHiperparameters.list_metrics, str_values)
        #AlgorithmsHiperparameters.id_general += 1
        
    
    def toFile(dataset_name, file_name):
        file = open(file_name, 'a')
        
        str_title = 'dataset\tid_general\thmm_loss\taccuracy\tjaccard_mi\tprecision\t'
        str_title += 'recall\tf_score\tprecision_mi\trecall_mi\tf_score_mi\tprecision_ma\trecall_ma\t'
        str_title += 'f_score_ma\tranking_loss\tcoverage\ttime\tclassifier\n'
        file.write(str_title)
        
        for string in AlgorithmsHiperparameters.list_metrics:
            string = dataset_name+'\t'+string
            file.write(string)
        
        AlgorithmsHiperparameters.list_metrics = []
        
        file.write('\n----------------------------------------------\n\n')
        file.close()