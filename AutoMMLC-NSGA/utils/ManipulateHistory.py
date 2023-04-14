import numpy as np

from pymoo.indicators.hv import Hypervolume


class ManipulateHistory:
    
    
    def get_hist_F(res):
        hist = res.history
    
        n_evals = []  # corresponding number of function evaluations\
        hist_F = []   # the objective space values in each generation

        for algorithm in hist:
            # store the number of function evaluations
            n_evals.append(algorithm.evaluator.n_eval)
        
            # retrieve the optimum from the algorithm
            opt = algorithm.opt
        
            # filter out only the feasible and append and objective space values
            feas = np.where(opt.get("feasible"))[0]
            hist_F.append(opt.get("F")[feas])
            
        return (n_evals, hist_F)
    
    
    def get_hypervolume(res):
        # Histórico de execução (número de avaliações e funções objetivos)
        n_evals, hist_F = ManipulateHistory.get_hist_F(res)
            
        # Hipervolume
        approx_ideal = res.F.min(axis=0)
        approx_nadir = res.F.max(axis=0)
        
        ref_point_2 = approx_nadir[1]
        
        metric = Hypervolume(
            # verificar o ponto de referência
            ref_point= np.array([1, ref_point_2]),
            norm_ref_point=False,
            zero_to_one=True,
            ideal=approx_ideal,
            nadir=approx_nadir
        )
    
        hv = [metric.do(_F) for _F in hist_F]
        
        return (n_evals, hist_F, hv)
    
    
    def get_mean_hist_F(hist_F):
        mean_1 = []
        mean_2 = []

        for f in hist_F:
            array = np.array(f)
            
            m1 = np.mean(array[:, 0])
            mean_1 = np.append(mean_1, m1)
            
            m2 = np.mean(array[:, 1])
            mean_2 = np.append(mean_2, m2)
            
        return (mean_1, mean_2)