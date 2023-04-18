import time

import numpy as np

import sklearn.metrics as metrics

# multi-label
from skmultilearn.adapt import MLkNN
from skmultilearn.adapt import MLTSVM
from skmultilearn.adapt import MLARAM
from skmultilearn.adapt import BRkNNaClassifier
from skmultilearn.adapt import BRkNNbClassifier

from skmultilearn.problem_transform import BinaryRelevance
from skmultilearn.problem_transform import ClassifierChain
from skmultilearn.problem_transform import LabelPowerset

# ensemble
from correcao_scikitmultlearn.rakeld import RakelD # correção
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

from utils.AlgorithmsHiperparameters import AlgorithmsHiperparameters

# process/thread
import multiprocessing 

from multiobjective.Pareto_Froint import Individual, FNDS



class RandomSearch:
    
    def __init__(self, x_train, x_test, y_train, y_test, n_thread, termination, limit_time, config, file_log):
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self.n_thread = n_thread
        self.termination = termination
        self.limit_time = limit_time
        self.config = config
        # ---------------------
        self.classifier_limit_time = 0 # número de classificadores que atingiram o tempo limite
        self.map_algs_objs = {} # algoritmo: objetivo1, objetivo2
        self.rep = 0
        # ---------------------
        self.file_log = file_log
        self.cont = 0 # para log

    
    def get_sl_algorithm(self):
        # escolhe algoritmo monorrótulo
        sl_algorithms = self.config.get_sl_algorithms()
        sl_alg = np.random.choice(sl_algorithms)
        
        # obtém a configuração do algoritmo
        config_sl_alg = self.config.get_sl_classifier_config()[sl_alg]
        
        # inicializa string com o classificador
        index = sl_alg.rfind('.')
        s = sl_alg[index+1:] + '('

        # escolhe os valores dos hiperparâmetros
        for key, values in config_sl_alg.items():
            choice = np.random.choice(values)
            
            if isinstance(choice, str):
                s +=  key + '=' + '\''+ choice + '\','
            else:
                s += key + '=' + str(choice) + ','
        
        # substitui a última vírgula
        s = s[:-1] + ')'
        
        return s
    

    def get_ml_algorithm(self):
        # escolhe algoritmo multirrótulo
        ml_algorithms = self.config.get_ml_algorithms()
        ml_alg = np.random.choice(ml_algorithms)
        
        # obtém a configuração do algoritmo
        config_ml_alg = self.config.get_ml_classifier_config()[ml_alg]
        
        # inicializa string com o classificador
        index = ml_alg.rfind('.')
        s = ml_alg[index+1:] + '('

        # escolhe os valores dos hiperparâmetros
        for key, values in config_ml_alg.items():
            if key == 'classifier' or key == 'base_classifier':
                s += key + '=' + self.get_sl_algorithm() + ','
            else:
                choice = np.random.choice(values)
                
                if isinstance(choice, str):
                    s +=  key + '=' + '\''+ choice + '\','
                else:
                    s += key + '=' + str(choice) + ','
        
        # substitui a última vírgula
        s = s[:-1] + ')'
        
        return s              
        
    
    def to_file(self):
        file = open(self.file_log, 'a')
        file.write(f'{str(self.cont)}\n')
        file.close()
        
    
    # task = classification -> paralelizável
    def task(self, s):                    
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
        
        return (s, prediction, f_score, time_execution)            
     

    def timeout(self, s):
        # variável de retorno: string do classificador, predição, f-score, tempo
        result = None
        
        str_classifier = s[0]
        
        if str_classifier in self.map_algs_objs.keys():
            f1 = self.map_algs_objs.get(str_classifier)[0] # f-score
            f2 = self.map_algs_objs.get(str_classifier)[1] # time
            self.rep += 1
            
            # aqui não é necessário informar prediction, pois a variável é utilizada
            # para calcular as métricas e para o classificador a métrica já está salva
            result = (str_classifier, None, f1, f2)
        
        # execução com limite de tempo
        else:
        
            try: 
                
                pool = multiprocessing.Pool(processes=1)
                # calcule os valores da função de maneira paralelizada e espere até terminar
                result = pool.apply_async(self.task, s).get(timeout=self.limit_time)
                pool.close()
    
            # caso o tempo limite exceda
            except multiprocessing.TimeoutError:
                
                pool.terminate()
                self.classifier_limit_time += 1
                
                result = (str_classifier, None, 0, self.limit_time) 
              
            # formato de result: (string, prediction, f1, f2) 
            self.map_algs_objs[str_classifier]= (result[2], result[3])
            
        # log
        # self.cont += 1 # contador
        # self.to_file() # armazena a avaliação corrente
         
        # retorno: str_classifier, f-score, time, prediction
        return result
    

    def execute_random_search(self):
        params = [[self.get_ml_algorithm()] for k in range(self.termination)]
        
        # pool de threads       
        with multiprocessing.pool.ThreadPool(self.n_thread) as pool:
            # faz a classificação com tempo limitado
            results = list(pool.map(self.timeout, params))        
        
        pool.close()
        pool.join()

        i = 0
        
        list_individuals = []
        for res in results:
            # formato de res = (k, string do classificador, f-score, time, prediction)
            s, prediction, f_score, time_execution = res
            
            # print(f'{i} - {s}, {f_score}, {time_execution}\n')
            
            # para os casos em que há classificação - não houve interrupção por limite de tempo
            if prediction is not None:
                # salva histórico de modelos treinados e suas métricas
                AlgorithmsHiperparameters.add_metrics(i, prediction, self.y_test, time_execution, s)
            
            individual = Individual(1-f_score, time_execution, s)
            list_individuals.append(individual)
            
            i += 1
        
        
        fnds = FNDS()
        pareto_froint = fnds.execute(list_individuals)    
        #fnds.plot_froint(list_individuals, pareto_froint, 'Objective Space', 'F Score', 'Time', None)
        
        return pareto_froint