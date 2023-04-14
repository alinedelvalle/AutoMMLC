import sys

import numpy as np

import pandas as pd

from scipy.sparse import lil_matrix

from configuration.Configuration import Configuration

from multiobjective.MLProblem import MLProblem
from multiobjective.MLSampling import MLSampling
from multiobjective.MLMutation import MLMutation
from multiobjective.DuplicateElimination import MLDuplicateEliminate

# termination 
from pymoo.util.termination.f_tol import MultiObjectiveSpaceToleranceTermination

from pymoo.optimize import minimize
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.factory import get_crossover

from utils.Graphic import Graphic
from utils.ManipulateHistory import ManipulateHistory
from utils.AlgorithmsHiperparameters import AlgorithmsHiperparameters

import pathlib
  
from pymoo.config import Config
    

if __name__ == "__main__":
    
    # se há parâmetros
    if len(sys.argv) > 1:
        name_dataset = sys.argv[1]              # nome do dataset
        k = int(sys.argv[2])                    # fold (um número de 0 a 9)
        len_poulation = int(sys.argv[3])        # tamanho da população
        number_generation = int(sys.argv[4])    # número de gerações
        n_thread = int(sys.argv[5])             # número de threads
        limit_time = int(sys.argv[6])           # tempo limite para execução do algoritmo multirrótulo
        project = str(sys.argv[7])              # folder do projeto
    else:
        name_dataset = 'flags'
        k = 0 # um número de 0 a 9
        len_poulation = 10
        number_generation = 10  
        n_thread = 4
        limit_time = 30
        project = None # definir caminho aqui
     
    Config.show_compile_hint = False
    
    # carrega datasets normalizados
    dfx = pd.read_csv('datasets/x_'+name_dataset+'.csv', sep=',')
    dfy = pd.read_csv('datasets/y_'+name_dataset+'.csv', sep=',')

    # carrega divisão dos folds
    folds = np.load('folds/'+name_dataset+'.npy', allow_pickle=True)
    
    config = Configuration(dfx.shape[1], dfy.shape[1])
    
    # para o fold k
    k = k * 2
    
    # obtém dados de treino e de teste
    index_train = folds[k]
    index_test = folds[k+1]

    x_train = lil_matrix(dfx.iloc[index_train])
    y_train = lil_matrix(dfy.iloc[index_train])
    x_test = lil_matrix(dfx.iloc[index_test])
    y_test = lil_matrix(dfy.iloc[index_test])
    
    problem = MLProblem(x_train, x_test, y_train, y_test, n_thread, limit_time, config, None)
    
    algorithm = NSGA2(
        pop_size=len_poulation,
        sampling=MLSampling(config),
        crossover=get_crossover("int_ux"),
        mutation=MLMutation(config, 0.05),
        eliminate_duplicates=MLDuplicateEliminate(config, problem),
    )

    termination = MultiObjectiveSpaceToleranceTermination(tol=0, n_max_gen=number_generation, n_last=10, nth_gen=1)

    res = minimize(
        problem,
        algorithm,
        termination,
        save_history=True,
        verbose=True
    )
    
    # imprime os resultados 
    print(f'Number of queries on the classifier and objectives map: {problem.rep}\n')
    print(f'Number of classifiers exceeding the time limit: {problem.classifier_limit_time}\n')
    
    # Gráfico
    # espaço de objetivos
    Graphic.plot_scatter(res.F[:, 0], res.F[:, 1], 'Objective Space', 'F Score', 'Time', 'test_files/'+name_dataset+'/ObjectiveSpace-'+str(k//2))
    
    # Gráfico    
    # hipervolume
    n_evals, hist_F, hv = ManipulateHistory.get_hypervolume(res)
    Graphic.plot_graphic(n_evals, hv, 'Convergence-Hypervolume', 'Evaluations', 'Hypervolume', 'test_files/'+name_dataset+'/Hypervolume-'+str(k//2))    
    
    # Prepara resultados para salvar em arquivo
    output_data = f'Number of queries on the classifier and objectives map: {problem.rep}\n'
    
    output_data += f'Number of classifiers exceeding the time limit: {problem.classifier_limit_time}\n'

    output_data += f'Execution time:{res.exec_time}\n'
      
    output_data += f"Best solution found:\n"
    for individual in res.X:
        output_data += f"{individual}\n"
     
    output_data += 'Classifiers:\n'
    for individual in res.X:
        string_classifier = problem.getMultiLabelClassifier(individual)
        output_data += f"{string_classifier}\n"
        
    output_data += f"Function value:\n{res.F}\n"
    
    list_fscore = []
    list_time = []
    for l in res.F:
        list_fscore = np.append(list_fscore, l[0])
        list_time = np.append(list_time, l[1])
    
    # f_score
    output_data += 'F_score\n['
    for i in range(len(list_fscore)-1):
        output_data += str(1 - list_fscore[i])+','
    output_data += str(1 - list_fscore[len(list_fscore)-1])+']\n'
     
    # time
    output_data += 'Time\n['
    for i in range(len(list_time)-1):
        output_data += str(list_time[i])+','
    output_data += str(list_time[len(list_time)-1])+']\n'
    
    output_data += 'Hypervolume:\n['
    for i in range(len(hv)-1):
        output_data += str(hv[i])+','
    output_data += str(hv[len(hv)-1])+']'
    
    output_data += '\nEvaluation:\n['
    for i in range(len(n_evals)-1):
        output_data += str(n_evals[i])+','
    output_data += str(n_evals[len(n_evals)-1])+']\n'
    
    output_data += 'History:\n'
    for array in hist_F:
        if len(array) == 1:
            output_data += str(array[0][0])+', '+str(array[0][len(array[0])-1])+'\n'
        else:
            for i in range(len(array)-1):
                output_data += str(array[i][0])+', '+str(array[i][len(array[i])-1])+', '
            output_data += str(array[i+1][0])+', '+str(array[i+1][len(array[i+1])-1])+'\n'     

    # Salva resultados
    output_path = pathlib.Path(f'test_files/{name_dataset}/results-{str(k//2)}.txt')
    output_path.write_text(output_data)
            
    # Salva medidas 
    AlgorithmsHiperparameters.toFile(name_dataset, 'test_files/'+name_dataset+'/metrics-'+str(k//2)+'.txt')