from pgmpy.models import BayesianModel 

from pgmpy.estimators import MaximumLikelihoodEstimator 

from pgmpy.inference import VariableElimination 

from pgmpy.factors.discrete import TabularCPD 

import numpy as np 

# Define the structure of the Bayesian network 

model = BayesianModel([('C', 'S'), ('D', 'S')]) 

# Define the conditional probability distributions (CPDs) 

cpd_c = TabularCPD('C', 2, [[0.5], [0.5]]) 

cpd_d = TabularCPD('D', 2, [[0.5], [0.5]]) 

cpd_s = TabularCPD('S', 2, [[0.8, 0.6, 0.6, 0.2], [0.2, 0.4, 0.4, 0.8]], 

 evidence=['C', 'D'], evidence_card=[2, 2]) 

# Add the CPDs to the model 

model.add_cpds(cpd_c, cpd_d, cpd_s)

# Create a Maximum Likelihood Estimator and fit the model to some data 

data = np.random.randint(low=0, high=2, size=(5000, 2)) 

mle = MaximumLikelihoodEstimator(model, data) 

model_fit = mle.fit() 

# Create a Variable Elimination object to perform inference 

infer = VariableElimination(model) 

# Perform inference on some observed evidence 

query = infer.query(['S'], evidence={'C': 1}) 

print(query)
