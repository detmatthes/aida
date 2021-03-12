import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC
#import xgboost
import sklearn.metrics as met
from sklearn.pipeline import Pipeline

# https://towardsdatascience.com/linear-regression-models-4a3d14b8d368
    

def init_df_model_metric_regression():
    df_model_metric = pd.DataFrame(columns=['Train Accuracy','CV Accuracy'])
    df_model_metric.index.name = "Model"
    return df_model_metric


def model_metric_regression(model, X_train, y_train, df_model_metric_regression, postname=None):    
    name = type(model).__name__
    if postname != None:
        name = name + '#' + postname
    print(f"Determine metrics of {model} ...")
    model.fit(X_train, y_train)
    acc = model.score(X_train, y_train)
    cv_accuracy = cross_val_score(model, X_train, y_train, cv=5).mean()
    print(f"Metrics of {model}: acc/cv_acc {acc: .5f}/{cv_accuracy: .5f}")
    df_model_metric_regression.at[name] = [acc,cv_accuracy]
    del model
    import gc
    gc.collect()
    return acc,cv_accuracy


def model_metric_LinearRegression(X_train, y_train, df_model_metric_regression):
    return model_metric_regression(LinearRegression(), X_train, y_train, df_model_metric_regression)

def model_metric_DecisionTreeRegressor(X_train, y_train, df_model_metric_regression):
    return model_metric_regression(DecisionTreeRegressor(), X_train, y_train, df_model_metric_regression)

def model_metric_RandomForestRegressor(X_train, y_train, df_model_metric_regression):
    return model_metric_regression(RandomForestRegressor(), X_train, y_train, df_model_metric_regression)

def model_metric_AdaBoostRegressor(X_train, y_train, df_model_metric_regression):
    return model_metric_regression(AdaBoostRegressor(), X_train, y_train, df_model_metric_regression)

def model_metric_GradientBoostingRegressor(X_train, y_train, df_model_metric_regression):
    return model_metric_regression(GradientBoostingRegressor(), X_train, y_train, df_model_metric_regression)



def init_df_search_best_param():
    df_model_metric = pd.DataFrame(columns=['Params','CV Scores Mean','CV Scores Std'])
    df_model_metric.index.name = "Model"
    return df_model_metric


def search_bestparam(model, param_grid, X_train, y_train, df_search_best_param, postname=None):
    name = type(model).__name__
    if name == 'Pipeline':
        name = str([modelname for (name, modelname) in model.steps])
    if postname != None:
        name = name + '#' + postname
    gridsearch = GridSearchCV(model, param_grid, cv=5, verbose=1, n_jobs=-1, scoring = 'r2')
    gridsearch.fit(X_train, y_train)
    bestmodel = gridsearch.best_estimator_
    cv_scores = cross_val_score(bestmodel, X_train, y_train, cv=5)
    cv_scores_mean = np.mean(cv_scores)
    cv_scores_std = np.std(cv_scores)
    print(f"Best param {gridsearch.best_params_}")
    print(f"Metrics of best {model}: cv-score(mean)/cv-score(std) {cv_scores_mean: .5f}/{cv_scores_std: .5f}")
    df_search_best_param.at[name] = [gridsearch.best_params_,cv_scores_mean, cv_scores_std]
    

def search_bestparam_LinearRegression(X_train, y_train, df_search_best_param):
    print(f"Search best params for LinearRegression ...")
    model = LinearRegression()
    print("Supported params", model.get_params())
    param_grid = {
          'normalize' : [True,False],
          'fit_intercept': [True,False]
      }
    search_bestparam(model, param_grid, X_train, y_train, df_search_best_param)


def search_bestparam_DecisionTreeRegressor(X_train, y_train, df_search_best_param):
    print(f"Search best params for DecisionTreeRegressor ...")
    model = DecisionTreeRegressor()
    print("Supported params", model.get_params())
    param_grid = {
          "criterion": ["mse", "mae"],
          "min_samples_split": [10, 20, 40],
          "max_depth": [2, 6, 8],
          "min_samples_leaf": [20, 40, 100],
          "max_leaf_nodes": [5, 20, 100]
      }
    search_bestparam(model, param_grid, X_train, y_train, df_search_best_param)


def search_bestparam_RandomForestRegressor(X_train, y_train, df_search_best_param):
    print(f"Search best params for RandomForestRegressor ...")
    model = RandomForestRegressor()
    print("Supported params", model.get_params())
    param_grid = {
              'n_estimators': [500, 700, 1000],
              'max_depth': [None, 1, 2, 3],
              'min_samples_split': [1, 2, 3]
      }
    search_bestparam(model, param_grid, X_train, y_train, df_search_best_param)
    
"""
def search_bestparam_AdaBoostRegressor(X_train, y_train, df_search_best_param):
    print(f"Search best params for AdaBoostRegressor ...")
    model = AdaBoostRegressor()
    print("Supported params", model.get_params())
    param_grid = {
              'n_estimators': (1, 2),
              'base_estimator__max_depth': (1, 2)
      }
    search_bestparam(model, param_grid, X_train, y_train, df_search_best_param)


def search_bestparam_GradientBoostingRegressor(X_train, y_train, df_search_best_param):
    print(f"Search best params for GradientBoostingRegressor ...")
    model = GradientBoostingRegressor()
    print("Supported params", model.get_params())
    param_grid = {
              'bootstrap': [True],
              'max_depth': [80, 90, 100, 110],
              'max_features': [2, 3],
              'min_samples_leaf': [3, 4, 5],
              'min_samples_split': [8, 10, 12],
              'n_estimators': [100, 200, 300, 1000]
      }
    search_bestparam(model, param_grid, X_train, y_train, df_search_best_param)
"""

"""    
def search_bestparam_LogisticRegression(X_train, y_train, df_search_best_param):
    print(f"Search best params for LogisticRegression ...")
    #https://www.dezyre.com/recipes/optimize-hyper-parameters-of-logistic-regression-model-using-grid-search-in-python
    X_scaled = StandardScaler().fit_transform(X_train)
    model = LogisticRegression()
    print("Supported params", model.get_params())
    param_grid = {
          #{'C': 1.0, 'class_weight': None, 'dual': False, 'fit_intercept': True, 'intercept_scaling': 1, 'l1_ratio': None, 'max_iter': 100, 'multi_class': 'auto', 'n_jobs': None, 'penalty': 'l2', 'random_state': None, 'solver': 'lbfgs', 'tol': 0.0001, 'verbose': 0, 'warm_start': False}
          "C": list(np.logspace(-3, 3, 7)),
          "penalty": ["l1","l2"],
          "solver": ["newton-cg", "lbfgs", "liblinear", "sag", "saga"],
          'class_weight': [{0:0.1,1:0.9}]
      }
    #https://mlfromscratch.com/gridsearch-keras-sklearn/#/
    search_bestparam(model, param_grid, X_scaled, y_train, df_search_best_param)
"""


def getRegressionModels():
    return [
        LinearRegression(),
        DecisionTreeRegressor(),
        RandomForestRegressor(),
        AdaBoostRegressor(),
        GradientBoostingRegressor()
    ]
