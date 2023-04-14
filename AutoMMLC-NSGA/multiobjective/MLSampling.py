import numpy as np

from pymoo.core.sampling import Sampling


class MLSampling(Sampling):
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.__map_algs_index = {}
        self.__map_index_algs = {0: 'ml_algorithm'}
        self.__len_X = 1
        

    # sorteia algoritmos multirrótulo mantendo uma proporção entre a ocorrência deles
    def get_algorithms(self, n_samples):
        n_algs = len(self.config.get_ml_algorithms())
        n = n_samples//n_algs
        algs = list(range(n_algs))

        ret = np.array(algs*n, dtype=int)

        if (n * n_algs != n_samples):
            aux = np.random.randint(0, n_algs, n_samples - (n * n_algs))
            ret = np.concatenate([ret, aux], axis=0)
    
        np.random.shuffle(ret)
        return ret
    
    
    # recebe o algoritmo e "sorteia" n_samples hiperparâmetros deste algoritmo
    def get_sample(self, alg, n_samples):
        # seleciona o dicionário de configurações do algoritmo single/multi rótulo
        if alg in self.config.get_ml_algorithms():
            dictionary = self.config.get_ml_classifier_config()[alg] 
        else:
            dictionary = self.config.get_sl_classifier_config()[alg]
        
        list_sample = np.array([], dtype=int)
        
        # para cada hiperparâmetro do algoritmo, sorteia n_samples valores 
        for key, value in dictionary.items(): 
            sample = np.random.randint(0, len(value), n_samples)   
            list_sample = np.append(list_sample, sample)
                
            self.__map_index_algs[self.__len_X] = (alg, key)
            self.__len_X = self.__len_X + 1
            
        # melhorar
        list_sample = list_sample.reshape(len(dictionary.keys()), n_samples)                
        X = np.column_stack(list_sample)
        
        return X
    
    
    def _do(self, problem, n_samples, **kwargs):       
        # a primeira informação do indivíduo é o algoritmo multirrótulo
        X = self.get_algorithms(n_samples)
        
        i = 1 # indice 0 é o algoritmo multirrotulo
        
        # para cada um dos algoritmos, seta os seus hiperparâmetros
        for alg in self.config.get_all_algorithms():           
            ret = self.get_sample(alg, n_samples)
            
            n = len(ret[0])
            self.__map_algs_index[alg] = (i, i + n - 1, n)            
            
            X = np.column_stack([X, ret])   
            
            i = i + n
         
        # Após definir X, seta o número de variáveis do problema    
        problem.n_var = len(X[0]) # self.__len_X
        problem.map_algs_index = self.__map_algs_index
        problem.map_index_algs = self.__map_index_algs
        
        return X