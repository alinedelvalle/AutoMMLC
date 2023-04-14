import time

import numpy as np

import sklearn.metrics as metrics

from pymoo.core.problem import Problem

from utils.AlgorithmsHiperparameters import AlgorithmsHiperparameters

# multi-label
from skmultilearn.adapt import MLkNN
from skmultilearn.adapt import MLTSVM
from skmultilearn.adapt import MLARAM
from skmultilearn.adapt import BRkNNaClassifier
from skmultilearn.adapt import BRkNNbClassifier

from skmultilearn.problem_transform import BinaryRelevance
from skmultilearn.problem_transform import ClassifierChain
from skmultilearn.problem_transform import LabelPowerset

# correção
from correcao_scikitmultlearn.rakeld import RakelD

from skmultilearn.ensemble import RakelO

# single-label
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import SGDClassifier

from scipy.sparse import lil_matrix

import multiprocessing


class MLProblem(Problem):
    
    
    def __init__(self, x_train, x_test, y_train, y_test, n_thread, limit_time, config, file_log, **kwargs):
        super().__init__(n_obj=2, n_constr=0, **kwargs) 
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self.n_thread = n_thread
        self.limit_time = limit_time
        self.config = config
        self.file_log = file_log
        self.ger = 0 # para log
        # algoritmo : 1º índice do hiperparâmetro no indivíduo, último índice, número de hiperparâmetros
        self.map_algs_index = {} 
        self.map_index_algs = {} # índice do indivíduo: algoritmo, hiperparâmetro
        self.map_algs_objs = {} # algoritmo: objetivo1, objetivo2
        self.rep = 0 # quantidade de vezes que o classificador foi reaproveitado do mapa
        self.classifier_limit_time = 0
    
    # obtém o nome da classe do classificador 
    def getClassClassifier(self, algorithm):
        index = algorithm.rfind('.')
        return algorithm[index+1:]

     
    # obtém uma string com a instanciação do classificador monorrótulo
    def getSingleLabelClassifier(self, algorithm, individual):
        first_index, last_index, n = self.map_algs_index[algorithm]
        
        config = self.config.get_sl_classifier_config()[algorithm]
            
        i = first_index # do indivíduo
        s = self.getClassClassifier(algorithm) + '('
        
        for variable in config.keys():
            all_values = config[variable]
            value = all_values[individual[i]]
                
            s = s + variable + "="
            
            if isinstance(value, str):
                s =  s + '\''+ value + '\','
            else:
                s =  s + str(value) + ','
                
            i = i + 1
        
        # substitui a última vírgula
        s = s[:-1] + ')'
            
        return s
    
    
    # obtém uma string com a instanciação do classificador multirrótulo
    def getMultiLabelClassifier(self, individual):
        index_algorithm = individual[0]
        algorithm = self.config.get_ml_algorithms()[index_algorithm]
            
        # do indivíduo
        first_index, last_index, _ = self.map_algs_index[algorithm] 
            
        config = self.config.get_ml_classifier_config()[algorithm]
            
        i = first_index # do indivíduo
        s = self.getClassClassifier(algorithm) + '('
        
        for variable in config.keys():
            all_values = config[variable]
            value = all_values[individual[i]]
            
            # single label 
            if variable == 'classifier' or variable == 'base_classifier':
                value = self.getSingleLabelClassifier(value, individual)
                s = s + variable + '=' + value + ','
            else:
                s = s + variable + "="
                
                if isinstance(value, str):
                    s =  s + '\''+ value + '\','
                else:
                    s =  s + str(value) + ','
                
            i = i + 1       
        
        # substitui a última vírgula
        s = s[:-1] + ')'
        
        return s    
    
    
    def to_file_ger(self):
        file = open(self.file_log, 'a')
        file.write(f'Geração: {str(self.ger)}\n')
        file.close()
        
        
    def to_file(self, s, f1, f2):
        file = open(self.file_log, 'a')
        file.write(f'{s} - {f1} - {f2}\n')
        file.close()
    
    
    # função de avaliação - fitness
    # processo - não compartilha memória
    def my_eval(self, s):
        classifier = eval(s)
        
        try:
            # treina modelo
            start_time = time.time()
            classifier.fit(self.x_train, self.y_train)
            end_time = time.time()
                    
            # testa o modelo
            prediction = classifier.predict(self.x_test)
                    
        except AttributeError:
            # Ocorre AttributeError ao utilizar o algoritmo MLARAM
            # Para utilizá-lo, é preciso converter o dataset para numpy.ndarray  
                    
            # Treina modelo
            start_time = time.time()
            classifier.fit(self.x_train.toarray(), self.y_train)
            end_time = time.time()
                    
            # testa o modelo
            prediction = classifier.predict(self.x_test.toarray())
                    
        except:  
            print('------------------------------------------------------- Erro -------------------------------------------------------')
            print(s)
            print('------------------------------------------------------- Erro -------------------------------------------------------')
                    
        time_execution = end_time - start_time
                
        f_score = metrics.f1_score(self.y_test, prediction, average='samples')
                 
        # objetivos
        f1 = 1 - f_score
        f2 = time_execution
            
        # print(f'---{s} - {f1} - {f2}---')
        
        return (s, prediction, f1, f2) 
    
    
    def timeout(self, s):
        result = None
        
        # formato de s: [string do cliassificador]
        str_classifier = s[0]
        
        if str_classifier in self.map_algs_objs.keys():
            f1 = self.map_algs_objs.get(str_classifier)[0] # 1 - f1-score
            f2 = self.map_algs_objs.get(str_classifier)[1] # time
            self.rep += 1
            
            #print('Entrou: {}, {}\n'.format(f1, f2))
            
            # aqui não é necessário informar prediction, pois a variável é utilizada
            # para calcular as métricas e o classificador a métrica já está salva
            result = (str_classifier, None, f1, f2)
        
        # execução com limite de tempo
        else:
        
            try: 
                
                pool = multiprocessing.Pool(processes=1)
                # calcule os valores da função de maneira paralelizada e espere até terminar
                result = pool.apply_async(self.my_eval, s).get(timeout=self.limit_time)
    
            # caso o tempo limite exceda
            except multiprocessing.TimeoutError:
                
                pool.terminate()
                self.classifier_limit_time += 1
                
                # returno: string do classificador, predição, objetivos - 1 para f-score e tempo limite
                result = (str_classifier, None, 1, self.limit_time) 
              
            # formato de result: (string, prediction, f1, f2) 
            self.map_algs_objs[str_classifier]= (result[2], result[3])
                
        return result
        
    
    def _evaluate(self, X, out, *args, **kwargs):       
        # prepara os parâmetros para o pool
        params = [[self.getMultiLabelClassifier(X[k])] for k in range(len(X))]
        
        # um pool de threads       
        with multiprocessing.pool.ThreadPool(self.n_thread) as pool:
            # faz a classificação com tempo limitado
            results = list(pool.map(self.timeout, params))
        
        pool.close()
        
        F = []
        for res in results:
            s = res[0]
            prediction = res[1]
            f_score = res[2] # f-score
            time = res[3] # time
            
            F.append([f_score, time]) 
            
            # para os casos em que não há classificação - limite de tempo
            if prediction is not None:
                # salva histórico de modelos treinados e suas métricas
                AlgorithmsHiperparameters.add_metrics(prediction, self.y_test, time, s)

        objectives = np.array(F)
        
        # store the function values and return them.
        out["F"] = objectives
        
        # log
        # self.ger += 1 # geração
        # self.to_file_ger() # armazena a geração corrente