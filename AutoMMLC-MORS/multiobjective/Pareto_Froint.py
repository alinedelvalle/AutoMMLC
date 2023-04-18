import matplotlib.pyplot as plt

import numpy as np


class Individual:
    
    def __init__(self, obj1, obj2, results):
        self.obj1 = obj1
        self.obj2 = obj2
        self.results = results
        
        
class Point:
    
    def __init__(self, individual):
        self.individual = individual
        self.__obj1 = self.individual.obj1
        self.__obj2 = self.individual.obj2
        self.n = 0
        self.rank = 0
        self.S = []
        
    def dominate(self, pto):
        flag = False
        
        if (self.__obj1 <= pto.__obj1 and self.__obj2 <= pto.__obj2):
            if (self.__obj1 < pto.__obj1 or self.__obj2 < pto.__obj2):
                flag = True
                
        return flag
    
    
    def __str__(self):
        return str(self.__obj1)+', '+str(self.__obj2)
        

# fast non dominated sort        
class FNDS:
    
    def execute(self, list_individuals):
        list_points = []
        
        # cria uma lista de pontos
        for ind in list_individuals:
            pto = Point(ind)
            list_points.append(pto)
        
        first_froint = []
        
        for i in range(len(list_points)):
            p = list_points[i]
            
            for j in range(len(list_points)):
                
                if i != j:
                    q = list_points[j]

                    if p.dominate(q):
                        p.S.append(q)
                    elif q.dominate(p):
                        p.n += 1

            if p.n == 0:
                p.rank = 1
                first_froint.append(p)
        
        return first_froint

        
    def plot_froint(self, list_individual, list_pareto, title, xlabel, ylabel, file_name):
        x_all = []
        y_all = []
        for ind in list_individual:
            x_all.append(ind.obj1)
            y_all.append(ind.obj2)
            
        x = []
        y = []
        for pto in list_pareto:
            ind = pto.individual
            x.append(ind.obj1)
            y.append(ind.obj2)
            
        plt.figure(figsize=(7, 5))
        plt.scatter(x_all, y_all, s=30, facecolors='none', edgecolors='gray')
        plt.scatter(x, y, s=30, facecolors='none', edgecolors='red')
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        #plt.savefig(file_name)
        
    
# teste dominância
if __name__ == '__main__':
    a = Individual(1, 5, None)
    b = Individual(2, 3, None)
    c = Individual(4, 1, None)
    d = Individual(3, 4, None)
    e = Individual(4, 3, None)
    f = Individual(5, 5, None)
    
    #pta = Point(a)
    #ptb = Point(b)
    #ptc = Point(c)
    #ptd = Point(d)
    #pte = Point(e)
    #ptf = Point(f)
    #print(pte.dominate(ptf))
    
    list_ind = [a, b, c, d, e, f]

    froint = FNDS().execute(list_ind)
    
    for pto in froint:
        print(pto)