import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.pipeline import Pipeline
#import xgboost
import sklearn.metrics as met
import lightgbm as lgb


def init_df_model_metric_classification():
    df_model_metric = pd.DataFrame(columns=['CV Accuracy','CV logloss','CV precision','CV recall','CV TP','CV TN','CV FP','CV FN'])
    df_model_metric.index.name = "Model"
    return df_model_metric


def model_metric_classification(model, X_train, y_train, df_model_metric, postname=None):    
    name = type(model).__name__
    if name == 'Pipeline':
        name = str([modelname for (name, modelname) in model.steps])
    if postname != None:
        name = name + '#' + postname
    print(f"Determine metrics of {model} ...")
    model.fit(X_train, y_train)
    #score = model.score(X, y)
    
    #y_pred = model.predict(X_train)
    #y_prob = model.predict_proba(X_train)
    y_pred = cross_val_predict(model, X_train, y_train, cv=5)
    cm = met.confusion_matrix(y_train, y_pred)
    TruePositiv = cm[0][0]
    TrueNegativ = cm[1][1]
    FalsePositiv = cm[0][1]
    FalseNegativ = cm[1][0]
    #acc = met.accuracy_score(y_train, y_pred) * 100
    #logloss = met.log_loss(y_train, y_prob)
    cv_accuracy = cross_val_score(model, X_train, y_train, cv=5).mean()
    cv_logloss = -1*cross_val_score(model, X_train, y_train, cv=5, scoring='neg_log_loss').mean()
    cv_precision = cross_val_score(model, X_train, y_train, cv=5, scoring='precision').mean()
    cv_recall = cross_val_score(model, X_train, y_train, cv=5, scoring='recall').mean()
    #print("cv_accuracy",cv_accuracy)
    #print("cv_logloss",cv_logloss)
    #print("cv_precision",cv_precision)
    #print("cv_recall",cv_recall)
    #c = cross_val_score(model, X_train, y_train, cv=5, scoring=['accuracy','neg_log_loss','precision','recall'])
    #print("c",c)
    #print(f"Metrics of {model}: acc/logloss/cv_acc/cv_logloss {acc: .5f}/{logloss: .5f}/{cv_accuracy: .5f}/{cv_logloss: .5f}")
    df_model_metric.at[name] = [cv_accuracy,cv_logloss,cv_precision,cv_recall,TruePositiv,TrueNegativ,FalsePositiv,FalseNegativ]
    del model
    import gc
    gc.collect()
    return cv_accuracy,cv_logloss

def model_metric_LogisticRegression(X_train, y_train, df_model_metric):
    pipe = Pipeline([('scale', StandardScaler()), ('lr', LogisticRegression())])
    return model_metric_classification(pipe, X_train, y_train, df_model_metric)

def model_metric_DecisionTreeClassifier(X_train, y_train, df_model_metric):
    return model_metric_classification(DecisionTreeClassifier(), X_train, y_train, df_model_metric)

def model_metric_RandomForestClassifier(X_train, y_train, df_model_metric):
    return model_metric_classification(RandomForestClassifier(), X_train, y_train, df_model_metric)

def model_metric_SVC(X_train, y_train, df_model_metric):
    pipe = Pipeline([('scale', StandardScaler()), ('lr', SVC(probability=True))])
    return model_metric_classification(pipe, X_train, y_train, df_model_metric)

def model_metric_AdaBoostClassifier(X_train, y_train, df_model_metric):
    return model_metric_classification(AdaBoostClassifier(), X_train, y_train, df_model_metric)

def model_metric_GradientBoostingClassifier(X_train, y_train, df_model_metric):
    return model_metric_classification(GradientBoostingClassifier(), X_train, y_train, df_model_metric)

def model_metric_LGBMClassifier(X_train, y_train, df_model_metric):
    return model_metric_classification(lgb.LGBMClassifier(), X_train, y_train, df_model_metric)



def init_df_search_best_param():
    df_model_metric = pd.DataFrame(columns=['Params','CV Accuracy','CV logloss', 'CV precision', 'CV recall','CV TP','CV TN','CV FP','CV FN'])
    df_model_metric.index.name = "Model"
    return df_model_metric


def search_bestparam(model, param_grid, X, y, df_search_best_param, postname=None, scoring=None):
    name = type(model).__name__
    if postname != None:
        name = name + '#' + postname
    #gridsearch = GridSearchCV(model, param_grid, cv=5, verbose=1, n_jobs=-1, refit='accuracy', scoring = ['accuracy','neg_log_loss'])
    #gridsearch = GridSearchCV(model, param_grid, cv=5, verbose=1, n_jobs=-1, refit = 'recall', scoring = ['recall','accuracy'])
    gridsearch = GridSearchCV(model, param_grid, cv=5, verbose=1, n_jobs=-1, scoring=scoring)
    
    gridsearch.fit(X, y)
    bestmodel = gridsearch.best_estimator_
    cv_accuracy = cross_val_score(bestmodel, X, y, cv=5).mean()
    
    y_pred = cross_val_predict(bestmodel, X, y, cv=5)
    cm = met.confusion_matrix(y, y_pred)
    TruePositiv = cm[0][0]
    TrueNegativ = cm[1][1]
    FalsePositiv = cm[0][1]
    FalseNegativ = cm[1][0]
    cv_logloss = -1*cross_val_score(bestmodel, X, y, cv=5, scoring='neg_log_loss').mean()
    cv_precision = cross_val_score(bestmodel, X, y, cv=5, scoring='precision').mean()
    cv_recall = cross_val_score(bestmodel, X, y, cv=5, scoring='recall').mean()
    print(f"Best param {gridsearch.best_params_}")
    print(f"Metrics of best {model}: cv-acc/cv-logloss {cv_accuracy: .5f}/{cv_logloss: .5f}")
    df_search_best_param.at[name] = [gridsearch.best_params_,cv_accuracy,cv_logloss,cv_precision,cv_recall,TruePositiv,TrueNegativ,FalsePositiv,FalseNegativ]
    


def search_bestparam_DecisionTreeClassifier(X, y, df_search_best_param):
    print(f"Search best params for DecisionTreeClassifier ...")
    from sklearn.tree import DecisionTreeClassifier
    model = DecisionTreeClassifier()
    print("Supported params", model.get_params())
    param_grid = {
          #{'ccp_alpha': 0.0, 'class_weight': None, 'criterion': 'gini', 'max_depth': None, 'max_features': None, 'max_leaf_nodes': None, 'min_impurity_decrease': 0.0, 'min_impurity_split': None, 'min_samples_leaf': 1, 'min_samples_split': 2, 'min_weight_fraction_leaf': 0.0, 'presort': 'deprecated', 'random_state': None, 'splitter': 'best'}
          "criterion": ["gini", "entropy"],
          "splitter": ["best", "random"],
          'max_depth': [None]+list(range(1,21)), 
          "max_features": ["auto", "sqrt", "log2"],
          'class_weight': [{0:0.1,1:0.9}, {0:0.2,1:0.8}, {0:0.3,1:0.7}, {0:0.4,1:0.6}, {0:0.5,1:0.5}, {0:0.6,1:0.4}, {0:0.7,1:0.3}, {0:0.8,1:0.2}, {0:0.9,1:0.1}]
      }
    """
    # draw score lines for each param
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(nrows=len(param_grid))
    for key in param_grid.keys():
        grid_part = { key : param_grid[key] }
        print("evaluate", grid_part)
        gridsearch = GridSearchCV(model, grid_part, cv=2, verbose=1, n_jobs=-1)
        gridsearch.fit(X, y)
        print("gridsearch.cv_results_['mean_test_score']", gridsearch.cv_results_['mean_test_score'])   
    """
    search_bestparam(model, param_grid, X, y, df_search_best_param)

def search_bestparam_RandomForestClassifier(X, y, df_search_best_param):
    print(f"Search best params for RandomForestClassifier ...")
    model = RandomForestClassifier()
    print("Supported params", model.get_params())
    param_grid = {
          "criterion": ["gini", "entropy"],
          'class_weight': [{0:0.1,1:0.9}, {0:0.2,1:0.8}, {0:0.3,1:0.7}, {0:0.4,1:0.6}, {0:0.5,1:0.5}, {0:0.6,1:0.4}, {0:0.7,1:0.3}, {0:0.8,1:0.2}, {0:0.9,1:0.1}],
          #"class_weight": ["balanced", "balanced_subsample"],
          # Number of trees in random forest
          #"n_estimators" : [int(x) for x in np.linspace(start = 100, stop = 20000, num = 51)],
          # Number of features to consider at every split
          #"max_features" : ['auto', 'sqrt', 'log2'],
          "max_features" : ['auto', 'sqrt'],
          # Maximum number of levels in tree
          #"max_depth" : [None]+[int(x) for x in np.linspace(10, 110, num = 11)],
          # Minimum number of samples required to split a node
          "min_samples_split" : [2, 5, 10, 15],
          # Minimum number of samples required at each leaf node
          "min_samples_leaf" : [1, 2, 4, 8],
          # Method of selecting samples for training each tree
          #"bootstrap" : [True, False]                    
          #"bootstrap": [True, False],
          #"warm_start": [True, False],
          #"criterion": ["gini", "entropy"],
          #"class_weight": ["balanced", "balanced_subsample"],
          #'max_depth': [None]+list(range(1,51)), 
          #"max_features": ["auto", "sqrt", "log2"],
          #"n_estimators": list(range(250)),
          #"max_leaf_nodes": list(range(100)),
          #"min_samples_split": np.arange(10,100,5),
          #"max_samples": np.arange(0,1.1,0.1),
          #"min_samples_leaf": np.arange(2,20,2)
      }
    search_bestparam(model, param_grid, X, y, df_search_best_param)

    
def search_bestparam_LogisticRegression(X, y, df_search_best_param):
    print(f"Search best params for LogisticRegression ...")
    #https://www.dezyre.com/recipes/optimize-hyper-parameters-of-logistic-regression-model-using-grid-search-in-python
    X_scaled = StandardScaler().fit_transform(X)
    model = LogisticRegression()
    print("Supported params", model.get_params())
    param_grid = {
          #{'C': 1.0, 'class_weight': None, 'dual': False, 'fit_intercept': True, 'intercept_scaling': 1, 'l1_ratio': None, 'max_iter': 100, 'multi_class': 'auto', 'n_jobs': None, 'penalty': 'l2', 'random_state': None, 'solver': 'lbfgs', 'tol': 0.0001, 'verbose': 0, 'warm_start': False}
          "C": list(np.logspace(-3, 3, 7)),
          "penalty": ["l1","l2"],
          "solver": ["newton-cg", "lbfgs", "liblinear", "sag", "saga"],
          'class_weight': [{0:0.1,1:0.9}, {0:0.2,1:0.8}, {0:0.3,1:0.7}, {0:0.4,1:0.6}, {0:0.5,1:0.5}, {0:0.6,1:0.4}, {0:0.7,1:0.3}, {0:0.8,1:0.2}, {0:0.9,1:0.1}]
          #'class_weight': ['None', 'balanced']
      }
    #https://mlfromscratch.com/gridsearch-keras-sklearn/#/
    search_bestparam(model, param_grid, X_scaled, y, df_search_best_param)


def search_bestparam_SVC(X, y, df_search_best_param):
    print(f"Search best params for SVC ...")
    X_scaled = StandardScaler().fit_transform(X)
    model = SVC(probability=True)
    print("Supported params", model.get_params())
    param_grid_2 = {
          'kernel': ['rbf'],
          "C": list(np.logspace(-3, 1, 10)),
          "gamma": ["scale", "auto"],
          'class_weight': [{0:0.1,1:0.9}, {0:0.2,1:0.8}, {0:0.3,1:0.7}, {0:0.4,1:0.6}, {0:0.5,1:0.5}, {0:0.6,1:0.4}, {0:0.7,1:0.3}, {0:0.8,1:0.2}, {0:0.9,1:0.1}],
      }
    search_bestparam(model, param_grid_2, X_scaled, y, df_search_best_param, postname='rbf')
    param_grid_3 = {
          'kernel': ["poly"],
          "C": list(np.logspace(-3, 1, 10)),
          "gamma": ["scale", "auto"],
          'class_weight': [{0:0.1,1:0.9}, {0:0.2,1:0.8}, {0:0.3,1:0.7}, {0:0.4,1:0.6}, {0:0.5,1:0.5}, {0:0.6,1:0.4}, {0:0.7,1:0.3}, {0:0.8,1:0.2}, {0:0.9,1:0.1}],
      }
    search_bestparam(model, param_grid_3, X_scaled, y, df_search_best_param, postname='poly')
    param_grid_linear = {
          'kernel': ['linear'],
          "C": list(np.logspace(-3, 1, 10)),
          'class_weight': [{0:0.1,1:0.9}, {0:0.2,1:0.8}, {0:0.3,1:0.7}, {0:0.4,1:0.6}, {0:0.5,1:0.5}, {0:0.6,1:0.4}, {0:0.7,1:0.3}, {0:0.8,1:0.2}, {0:0.9,1:0.1}],
      }
    search_bestparam(model, param_grid_linear, X_scaled, y, df_search_best_param, postname='linear')
   


def search_bestparam_LinearSVC(X, y, df_search_best_param):
    print(f"Search best params for LinearSVC ...")
    X_scaled = StandardScaler().fit_transform(X)
    model = LinearSVC(probability=True)
    print("Supported params", model.get_params())
    param_grid = {
          'penalty': ['l1', 'l2'],
             'class_weight': [{0:0.1,1:0.9}, {0:0.2,1:0.8}, {0:0.3,1:0.7}, {0:0.4,1:0.6}, {0:0.5,1:0.5}, {0:0.6,1:0.4}, {0:0.7,1:0.3}, {0:0.8,1:0.2}, {0:0.9,1:0.1}],
             'dual': [True, False],
             'loss': ['hinge', 'squared_hinge'],
             'multi_class': ['ovr', 'crammer_singer'],
             'fit_intercept': [True, False],
             'max_iter': np.arange(100,5050,50),
             'C': list(np.logspace(-3, 3, 20)),
             'intercept_scaling': list(np.arange(10,110,10)/100)
      }
    search_bestparam(model, param_grid, X_scaled, y, df_search_best_param)


# https://www.programcreek.com/python/example/83258/sklearn.ensemble.AdaBoostClassifier
def search_bestparam_AdaBoostClassifier(X, y, df_search_best_param):
    print(f"Search best params for AdaBoostClassifier ...")
    model = AdaBoostClassifier()
    print("Supported params", model.get_params())
    param_grid = {
          'n_estimators': [1,10,100,1000],
          'algorithm': ['SAMME', 'SAMME.R']
      }
    search_bestparam(model, param_grid, X, y, df_search_best_param)
    
    
    
def search_bestparam_GradientBoostingClassifier(X, y, df_search_best_param):
    print(f"Search best params for GradientBoostingClassifier ...")
    model = GradientBoostingClassifier()
    print("Supported params", model.get_params())
    param_grid = {
          'n_estimators': [1,10,100,1000],
          'learning_rate' : [0.001,0.01,0.05,0.1,0.5],
          'subsample' : [0.1,0.5,1.0],
          'max_depth': [1,3,5,10,20,50,100]
      }
    search_bestparam(model, param_grid, X, y, df_search_best_param)


def search_bestparam_LGBMClassifier(X, y, df_search_best_param):
    print(f"Search best params for LGBMClassifier ...")
    model = lgb.LGBMClassifier()
    print("Supported params", model.get_params())
    param_grid = {
          'num_leaves': [20, 30, 40],
          'bagging_fraction': (0.5, 0.8),
            'bagging_frequency': (5, 8),
            'feature_fraction': (0.5, 0.8),
            'max_depth': (10, 13),
            'min_data_in_leaf': (90, 120)
      }
    search_bestparam(model, param_grid, X, y, df_search_best_param)


def getClassificationModels():
    return [DecisionTreeRegressor()]


def model_metrics(X_train, y_train):
    df_model_metric = init_df_model_metric_classification()
    model_metric_LogisticRegression(X_train, y_train, df_model_metric)
    model_metric_DecisionTreeClassifier(X_train, y_train, df_model_metric)
    model_metric_RandomForestClassifier(X_train, y_train, df_model_metric)
    model_metric_AdaBoostClassifier(X_train, y_train, df_model_metric)
    model_metric_GradientBoostingClassifier(X_train, y_train, df_model_metric)
    model_metric_SVC(X_train, y_train, df_model_metric)
    return df_model_metric


def model_best_param(X_train, y_train):
    df_search_best_param = init_df_search_best_param()
    search_bestparam_LogisticRegression(X_train, y_train, df_search_best_param)
    search_bestparam_DecisionTreeClassifier(X_train, y_train, df_search_best_param)
    search_bestparam_RandomForestClassifier(X_train, y_train, df_search_best_param)
    search_bestparam_AdaBoostClassifier(X_train, y_train, df_search_best_param)
    search_bestparam_GradientBoostingClassifier(X_train, y_train, df_search_best_param)
    search_bestparam_SVC(X_train, y_train, df_search_best_param)
    return df_search_best_param

