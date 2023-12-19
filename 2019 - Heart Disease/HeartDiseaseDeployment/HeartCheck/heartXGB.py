import os
import sys
import pandas as pd
import numpy as np
import itertools
import warnings
import pickle
from sklearn.preprocessing import RobustScaler
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from tpot_metrics import balanced_accuracy_score

def predict(data):
    X_test = [[data['cp'],data['thalach'],data['exang'],data['oldpeak'],data['slope'],data['ca'],data['thal']]]

    for pipe_parameters in pipelines:
        pipeline = []
        for component in pipeline_components:
            if component in pipe_parameters:
                args = pipe_parameters[component]
                pipeline.append(component(**args))
            else:
                pipeline.append(component())

        # make pipeline
        clf = make_pipeline(*pipeline)

        # load the data frame
        if os.path.exists(newFile):
            dataFrame = pd.read_csv(newFile, compression='gzip', sep='\t')
        else:
            dataFrame = pd.read_csv(origFile, compression='gzip', sep='\t')

        # save the model to disk
        if not os.path.exists(modelFile):
            features = dataFrame.drop('target', axis=1).values
            labels = dataFrame['target'].values
            clf.fit(features, labels)
            pickle.dump(clf, open(modelFile, 'wb'))
        
        # load the model from disk
        loaded_model = pickle.load(open(modelFile, 'rb'))

        # predict the label
        pred = loaded_model.predict(X_test)

        # get the probability of the prediction
        prob = loaded_model.predict_proba(X_test)

        # set the result json to return
        return '{"pred":' + str(pred[0]) + ',"prob":' + str(prob[0][0]) + ',"count":' + str(dataFrame.shape[0]) + '}'

def addToModel(data):
    if os.path.exists(modelFile):
        os.remove(modelFile)
    if os.path.exists(newFile):
        dataFrame = pd.read_csv(newFile, compression='gzip', sep='\t')
    else:
        dataFrame = pd.read_csv(origFile, compression='gzip', sep='\t')
    dataFrame = dataFrame.append(data, ignore_index=True)
    dataFrame = dataFrame.sample(frac=1).reset_index(drop=True)
    dataFrame.to_csv(newFile, sep='\t', compression='gzip', index=False)
    return '{"count":' + str(dataFrame.shape[0]) + '}'

def resetModel():
    if os.path.exists(modelFile):
        os.remove(modelFile)
    if os.path.exists(newFile):
        os.remove(newFile)
    dataFrame = pd.read_csv(origFile, compression='gzip', sep='\t')
    return '{"count":' + str(dataFrame.shape[0]) + '}'

pipeline_components = [RobustScaler, XGBClassifier]
pipeline_parameters = {}

n_estimators_values = [50]
learning_rate_values = [0.1]
gamma_values = [0.4]
max_depth_values = [3]
subsample_values = [0.2]
random_state = [324089]

all_param_combinations = itertools.product(n_estimators_values, learning_rate_values, gamma_values, max_depth_values, subsample_values, random_state)
pipeline_parameters[XGBClassifier] = \
   [{'n_estimators': n_estimators, 'learning_rate': learning_rate, 'gamma': gamma, 'max_depth': max_depth, 'subsample': subsample, 'seed': random_state, 'nthread': 1}
     for (n_estimators, learning_rate, gamma, max_depth, subsample, random_state) in all_param_combinations]

pipelines = [dict(zip(pipeline_parameters.keys(), list(parameter_combination)))
    for parameter_combination in itertools.product(*pipeline_parameters.values())]

origFile = 'PMLB/heart-c-select.tsv.gz'    
newFile = 'PMLB/heart-c-select-new.tsv.gz'
modelFile = 'PMLB/heart-c-select_xgb_model.pkl'

#predict({oper: 'predict', 'cp': 3, 'thalach': 150, 'exang': 0, 'oldpeak': 2.3, 'slope': 0, 'ca': 0, 'thal': 1})
#addToModel({'oper': 'addToModel', 'cp': 3, 'thalach': 150, 'exang': 0, 'oldpeak': 2.3, 'slope': 0, 'ca': 0, 'thal': 1, 'target': 1})
