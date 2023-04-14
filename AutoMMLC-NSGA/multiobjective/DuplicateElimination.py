from pymoo.core.duplicate import ElementwiseDuplicateElimination


class MLDuplicateEliminate(ElementwiseDuplicateElimination):
    
    def __init__(self, config, problem):
        super().__init__()
        self.config = config
        self.__problem = problem
        

    def is_equal(self, a, b):
        # first_index = 0, first_index_2 = 0
        flag = False
        
        a = a.X
        b = b.X
        
        # mesmo algoritmo multirrótulo
        if a[0] == b[0]: 
            flag = True
            
            # obtém algoritmo multirrótulo
            index_alg = a[0]
            algorithm =self.config.get_ml_algorithms()[index_alg]
            
            # obtém mapa de algoritmo x índices
            map_index = self.__problem.map_algs_index # (primeiro, último, n)
            
            # obtém o primeiro índice do hiperparâmetro do algoritmo no indivíduo
            first_index, _, _ = map_index[algorithm]
            i = first_index
            
            # obtém a configuração do indivíduo
            config = self.config.get_ml_classifier_config()[algorithm]
            
            # verifica igualdade dos hiperparâmetros dos algoritmos multirrótulo
            for variable in config:
                if a[i] != b[i]:
                    flag = False
                    break
                
                # verifica igualdade dos hiperparâmetros dos algoritmos monorrótulos
                if variable == 'classifier':
                    # obtém algoritmo monorrótulo
                    index_alg_2 = a[i] # a[i] == b[i]
                    algorithm_2 = self.config.get_sl_algorithms()[index_alg_2]
                    
                    # obtém o primeiro índice do hiperparâmetro do algoritmo no indivíduo
                    first_index_2, last_index_2, _ = map_index[algorithm_2]
                    for j in range(first_index_2, last_index_2 + 1):
                        if a[j] != b[j]:
                            flag = False
                            break
                        
                if flag == False:
                    break
                        
                i = i + 1
        
        return flag