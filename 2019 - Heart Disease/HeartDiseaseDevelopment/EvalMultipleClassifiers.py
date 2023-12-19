import sys
import pandas as pd
import numpy as np
import itertools
import datetime
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier # Assumes XGBoost v0.6
from sklearn.ensemble import RandomForestClassifier
from evaluate_model import evaluate_model

# Print start time
print(datetime.datetime.now())

# GBC
dataset = '../../../HeartTests/PMLB/heart-c-select.tsv.gz'

pipeline_components = [RobustScaler, GradientBoostingClassifier]
pipeline_parameters = {}

n_estimators_values = [10, 50, 100, 500]
min_impurity_decrease_values = np.arange(0., 0.005, 0.00025)
max_features_values = [0.1, 0.25, 0.5, 0.75, 'sqrt', 'log2', None]
learning_rate_values = [0.01, 0.1, 0.5, 1.0, 10.0, 50.0, 100.0]
loss_values = ['deviance', 'exponential']
random_state = [324089]

all_param_combinations = itertools.product(n_estimators_values, min_impurity_decrease_values, max_features_values, learning_rate_values, loss_values, random_state)
pipeline_parameters[GradientBoostingClassifier] = \
   [{'n_estimators': n_estimators, 'min_impurity_decrease': min_impurity_decrease, 'max_features': max_features, 'learning_rate': learning_rate, 'loss': loss, 'random_state': random_state}
     for (n_estimators, min_impurity_decrease, max_features, learning_rate, loss, random_state) in all_param_combinations]

evaluate_model(dataset, pipeline_components, pipeline_parameters)
print(datetime.datetime.now())

# XGB
dataset = '../../../HeartTests/PMLB/heart-c-select.tsv.gz'

pipeline_components = [RobustScaler, XGBClassifier]
pipeline_parameters = {}

n_estimators_values = [10, 50, 100, 500]
learning_rate_values = [0.01, 0.1, 0.5, 1.0, 10.0, 50.0, 100.0]
gamma_values = np.arange(0., 0.51, 0.05)
max_depth_values = [1, 2, 3, 4, 5, 10, 20, 50, None]
subsample_values = np.arange(0.0, 1.01, 0.1)
random_state = [324089]

all_param_combinations = itertools.product(n_estimators_values, learning_rate_values, gamma_values, max_depth_values, subsample_values, random_state)
pipeline_parameters[XGBClassifier] = \
   [{'n_estimators': n_estimators, 'learning_rate': learning_rate, 'gamma': gamma, 'max_depth': max_depth, 'subsample': subsample, 'seed': random_state, 'nthread': 1}
     for (n_estimators, learning_rate, gamma, max_depth, subsample, random_state) in all_param_combinations]

evaluate_model(dataset, pipeline_components, pipeline_parameters)
print(datetime.datetime.now())

# RandomForest
dataset = '../../../HeartTests/PMLB/heart-c-select.tsv.gz'

pipeline_components = [RobustScaler, RandomForestClassifier]
pipeline_parameters = {}

n_estimators_values = [10, 50, 100, 500]
min_impurity_decrease_values = np.arange(0., 0.005, 0.00025)
max_features_values = [0.1, 0.25, 0.5, 0.75, 'sqrt', 'log2', None]
criterion_values = ['gini', 'entropy']
random_state = [324089]

all_param_combinations = itertools.product(n_estimators_values, min_impurity_decrease_values, max_features_values, criterion_values, random_state)
pipeline_parameters[RandomForestClassifier] = \
   [{'n_estimators': n_estimators, 'min_impurity_decrease': min_impurity_decrease, 'max_features': max_features, 'criterion': criterion, 'random_state': random_state}
     for (n_estimators, min_impurity_decrease, max_features, criterion, random_state) in all_param_combinations]

evaluate_model(dataset, pipeline_components, pipeline_parameters)
print(datetime.datetime.now())
