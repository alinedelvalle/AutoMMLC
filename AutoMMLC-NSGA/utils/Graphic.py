import matplotlib.pyplot as plt

import numpy as np


class Graphic:
    
    
    def plot_graphic(x, y, title, xlabel, ylabel, file_name):
        plt.figure(figsize=(7, 5))
        plt.plot(x, y,  color='black', lw=0.7)
        #plt.scatter(x, y,  facecolor="none", edgecolor='black', marker="p")
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.savefig(file_name)
        
        
    def plot_lines(list_x, list_y, labels, title, xlabel, ylabel, file_name):
        plt.figure(figsize=(10, 7))
        
        for i in range(len(list_x)):
            
            plt.plot(list_x[i], list_y[i], lw=1, label=labels[i])
        
        plt.legend()
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.savefig(file_name) 
        plt.show()
        
        
    def __count_points(x, y):
        array = [[x[i], y[i]] for i in range(len(x))]
        unique, counts = np.unique(array, axis=0, return_counts=True)
        return (unique, counts)
    
        
    def plot_scatter(x, y, title, xlabel, ylabel, file_name):
        unique, counts = Graphic.__count_points(x, y)
        plt.figure(figsize=(7, 5))
        plt.scatter(unique[:, 0], unique[:, 1], s=30, facecolors='none', edgecolors='blue', linewidths=counts)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.savefig(file_name)
        
        
    def plot_mean_std(x, mean, std, title, xlabel, ylabel, file_name):
        plt.figure(figsize=(10, 7))
        #plt.errorbar(x, mean, std, linestyle='None') # , marker='^'
        plt.plot(x, mean)
        plt.fill_between(x, mean - std, mean + std, alpha=0.3)
        
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.savefig(file_name)
        

    def plot_bar(df, index_x, index_y, title, xlabel, ylabel, file_name):       
        df.plot(x=index_x, y=index_y, kind="bar", figsize=(14, 5), stacked=True)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.savefig(file_name)