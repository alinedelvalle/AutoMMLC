import sys

import time

import os
import tarfile

import pandas as pd

import numpy as np

from scipy.sparse import lil_matrix

from configuration.Configuration import Configuration

from multiobjective.RandomSearch import RandomSearch

from utils.Graphic import Graphic
from utils.AlgorithmsHiperparameters import AlgorithmsHiperparameters

from pymoo.factory import get_performance_indicator

import pathlib


if __name__ == '__main__':
    # se há parâmetros
    if len(sys.argv) > 1:
        name_dataset = sys.argv[1]      # nome do dataset
        k = int(sys.argv[2])            # fold (um número de 0 a 9)
        termination = int(sys.argv[3])  # número de soluções avaliadas
        n_thread = int(sys.argv[4])     # número de threads
        limit_time = int(sys.argv[5])   # tempo limite para execução do algoritmo multirrótulo (em segundos)
        project = str(sys.argv[6])      # folder do projeto
    else:
        name_dataset = 'emotions'
        k = 1 # um número de 0 a 9
        termination = 100
        n_thread = 3     
        limit_time = 30 # em segundos
        project = None # informar path para o projeto
    
    start_time = time.time()
        
    # carrega datasets normalizados
    dfx = pd.read_csv(project+'/datasets/x_'+name_dataset+'.csv', sep=',')
    dfy = pd.read_csv(project+'/datasets/y_'+name_dataset+'.csv', sep=',')

    # carrega divisão dos folds
    folds = np.load(project+'/folds/'+name_dataset+'.npy', allow_pickle=True)
    
    # para o fold k
    k = k * 2
    
    index_train = folds[k]
    index_test = folds[k+1]

    x_train = lil_matrix(dfx.iloc[index_train])
    y_train = lil_matrix(dfy.iloc[index_train])
    x_test = lil_matrix(dfx.iloc[index_test])
    y_test = lil_matrix(dfy.iloc[index_test])
       
    # configuration
    config = Configuration(x_test.shape[0], y_test.shape[1])
    
    # random search    
    rs = RandomSearch(x_train, x_test, y_train, y_test, n_thread, termination, limit_time, config, None)
    pareto_froint = rs.execute_random_search()
    
    end_time = time.time()
    time_execution = end_time - start_time  
    
    output_data = f'Number of queries on the classifier and objectives map: {rs.rep}\n'
    output_data += f'Number of classifiers exceeding the time limit: {rs.classifier_limit_time}\n'
    output_data += f'Execution time: {time_execution}\n'
    
    output_data += 'Classifiers:\n'
    for pto in pareto_froint:
        output_data += pto.individual.results+'\n'
     
    # na fronteira de pareto temos time e (1 - f-score)
    list_1_fscore = []
    output_data += 'F_score\n'
    for pto in pareto_froint:
        output_data += f'{1 - pto.individual.obj1} '
        list_1_fscore.append(pto.individual.obj1)
    output_data += '\n'

    list_time = []
    output_data += 'Time\n'
    for pto in pareto_froint:
        output_data += f'{pto.individual.obj2} '
        list_time.append(pto.individual.obj2)
    output_data += '\n'
    
    output_data +=  'Hypervolume:\n'
    ref_point = [1, 600]
    F = np.array([list_1_fscore, list_time]).transpose()    
    hv = get_performance_indicator("hv", ref_point=ref_point).do(F)
    output_data += f'{hv}'
    
    
    # Medidas
    AlgorithmsHiperparameters.toFile(name_dataset, project+'/test_files/'+name_dataset+'/metrics-'+str(k//2)+'.txt')

    #print(output_data)
    output_path = pathlib.Path(f'{project}/test_files/{name_dataset}/results-{str(k//2)}.txt')
    output_path.write_text(output_data)
    
    # Gráfico
    # espaço de objetivos
    Graphic.plot_scatter(list_1_fscore, list_time, 'Objective Space', 'F Score', 'Time', 'test_files/'+name_dataset+'/ObjectiveSpace-'+str(k//2))