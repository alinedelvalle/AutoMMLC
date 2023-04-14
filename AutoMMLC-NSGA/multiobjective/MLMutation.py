import numpy as np

from pymoo.core.mutation import Mutation


class MLMutation(Mutation):
    
    
    def __init__(self, config, prob):
        super().__init__() 
        self.prob = prob
        self.config = config
        
    
    def _do(self, problem, X, **kwargs):
        
        for individual in X:
            
            n_rand = np.random.rand()
            
            if (n_rand <= self.prob):
                
                index = np.random.randint(0, len(individual))
                
                if index == 0: # mutação do algoritmo multirrótulo
                    individual[index] = np.random.randint(0, len(self.config.get_ml_algorithms()))
                else:
                    map_index_algs = problem.map_index_algs
                    algorithm, hyperparameter = map_index_algs[index]
                    
                    # seleciona o dicionário de configurações do algoritmo single/multi rótulo
                    if algorithm in self.config.get_ml_algorithms():
                        dictionary = self.config.get_ml_classifier_config()[algorithm] 
                    else:
                        dictionary = self.config.get_sl_classifier_config()[algorithm]
                    
                    # obtém possíveis valores do hiperparâmetro
                    all_values = dictionary[hyperparameter]
                    
                     # se há opções para alterar o hiperparâmetro
                    if len(all_values) > 1:
                        # guarda o valor antigo do hiperparâmetro
                        old_value = all_values[individual[index]]
                        
                        # sorteia o índice de um destes valores
                        index_value = np.random.randint(0, len(all_values))
                        
                        # obtém o novo valor do hiperparâmetro
                        new_value = all_values[index_value]
                        
                        # enquanto os valores novo e antigo forem iguais, tenta encontrar um novo valor
                        while (new_value == old_value):
                            # sorteia o índice de um destes valores
                            index_value = np.random.randint(0, len(all_values))
                            new_value = all_values[index_value]
                        
                        # atualiza o indivíduo
                        individual[index] = index_value
                
        return X