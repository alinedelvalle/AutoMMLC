# AutoMMLC

In this repository, we make available the datasets resulting from the datasets pre-processing, the codes of the Automated Multi-objective Multi-label Classification (AutoMMLC) method with the Multi-objective Random Search (MORS) and Non-Dominated Sorting Genetic Algorithm II (NSGA-II) algorithms, and the Supplementary Material. Below we present the structure of the two projects and inform how they can be executed.

# AutoMMLC<sub>NSGA</sub>

Project Structure:
* **configuration**: the folder contains files with the search space definition and the *Configuration* class, which is responsible for obtaining the search space information.
* **datasets**: folder where pre-processed datasets are located. When executing the AutoML method, the dataset is obtained from this folder.
* **folds**: folder where the divisions of the folds by dataset are. For the *emotions* dataset, for example, the emotions.npy file contains the indices of the instances that belong to the folds for the folds.
* **multiobjective**: the folder contains classes that have been redefined from the Pymoo framework:
  * *MLSampling.py*: class responsible for obtaining individuals from the search space. The *_do()* method returns the population, where each individual represents a multi-label classification algorithm.
  *  *MLProblem.py*: class responsible for evaluating the individuals of the population and obtaining the values of the objectives of the multiobjective optimization. As the evaluation of each individual is independent (training and obtaining the evaluation measures), the evaluation can occur in parallel. The *_evaluate* method implements parallelism and returns the values of the individual's goals.
  *   *MLMutation.py*: class responsible for the mutation. The mutation occurs with a probability. Individuals have an altered gene with a valid search space value.
  *   *MLDuplicateEliminate.py*: class responsible for checking whether there are duplicated individuals in the population. The *is_equal(self, a, b)* method checks whether individuals *a* and *b* are equal. The class is used to eliminate duplicate individuals in the initial population.
* **teste_files**: files from the execution of the AutoML method are stored in this folder:
  * *metrics-k.txt*: contains the execution history (algorithms and evaluation measures). The file is stored in the folder corresponding to the dataset, and *k* specifies the fold.
  * *results-k.txt*: contains the result obtained by the AutoML method (Pareto frontier algorithms, f-score of each algorithm, the training time of each algorithm, number of solutions evaluated over the generations, and resulting hypervolume at the end of each generation). The file is stored in the folder corresponding to the dataset, and *k* specifies the fold.
* **utils**: the folder contains classes applicable to the project, such as the *AlgorithmsHiperparameters* class (it has methods to store and save the evaluated classifiers and the calculated evaluation measures).
* **main.py**: python file to execute the AutoML method. See below how the AutoML method with NSAG-II can be executed.

**How to run?** To run the *main.py*, it is necessary to inform the parameters:
* dataset - dataset name.
* k_fold - fold (a number from 0 to 9).
* population_size - population size.
* n_generation - number of generations.
* n_thread - number of threads.
* limit_time - time limit for running the multi-label algorithm (in seconds).
* project_path - project folder.

Thus, the command line for execution is:

```python main.py dataset k_fold population_size n_generation n_thread limit_time project_path```


# AutoMMLC<sub>MORS</sub>

Project Structure:
* **configuration**: the folder contains files with the search space definition and the *Configuration* class, which is responsible for obtaining the search space information.
* **datasets**: folder where pre-processed datasets are located. When executing the AutoML method, the dataset is obtained from this folder.
* **folds**: folder where the divisions of the folds by dataset are. For the *emotions* dataset, for example, the emotions.npy file contains the indices of the instances that belong to the folds for the folds.
* **Multiobjective**: the folder contains the implementation of the AutoML method with Random Search Multiobjective (*RandomSearch.py*). This algorithm selects *T* multi-label classification algorithms from the search space, trains the algorithms, and evaluates the trained models. As the training and evaluation of the models are independent for each algorithm, this occurs in parallel. Then, the algorithm finds the Pareto frontier with the values of the objectives (*training time* and *1 – f_score*). The *Pareto_Froint.py* file contains the classes needed to find the Pareto frontier.
* **teste_files**: files from the execution of the AutoML method are stored in this folder:
  * *metrics-k.txt*: contains the execution history (algorithms and evaluation measures). The file is stored in the folder corresponding to the dataset, and *k* specifies the fold.
  * *results-k.txt*: contains the result obtained by the AutoML method (Pareto frontier algorithms, f-score of each algorithm, the training time of each algorithm, number of solutions evaluated over the generations, and resulting hypervolume). The file is stored in the folder corresponding to the dataset, and *k* specifies the fold.
* **utils**: the folder contains classes applicable to the project, such as the *AlgorithmsHiperparameters* class (it has methods to store and save the evaluated classifiers and the calculated evaluation measures).
* **main.py**: python file to execute the AutoML method. See below how the AutoML method with MORS can be executed.

**How to run?** To run the *main.py*, it is necessary to inform the parameters:
* dataset - dataset name.
* k_fold - fold (a number from 0 to 9).
* termination - number of evaluated solutions.
* n_thread - number of threads.
* limit_time - time limit for running the multi-label algorithm (in seconds).
* project_path - project folder.

Thus, the command line for execution is:

```python main.py dataset k_fold termination n_thread limit_time project_path```
