# example: Sklearn Dataset

import pandas as pd

from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()

df = pd.DataFrame(data.data, columns=data.feature_names)
targetColumn = 'target'
df['target'] = pd.Series(data.target)

print(df[targetColumn].isna().sum())

import aidalib.feature_preparation as feature_preparation

X_train, X_test, y_train, y_test = feature_preparation.helper_split(df, targetColumn, test_size=0.2, random_state=99)


import aidalib.feature_reduction as feature_reduction


df_feature_importances = feature_reduction.determine_feature_importance(X_train, y_train)
X_train = feature_reduction.drop_unimportant_features(X_train, df_feature_importances)
X_test = feature_reduction.drop_unimportant_features(X_test, df_feature_importances)

columns_with_low_correlated_features_to_target, _ = feature_reduction.determine_low_correlated_features_to_target(X_train, y_train)
X_train.drop(columns=columns_with_low_correlated_features_to_target, inplace=True)
X_test.drop(columns=columns_with_low_correlated_features_to_target, inplace=True)

columns_with_strong_correlation_between, _ = feature_reduction.determine_strong_correlated_features(X_train, y_train)
X_train.drop(columns=columns_with_strong_correlation_between, inplace=True)
X_test.drop(columns=columns_with_strong_correlation_between, inplace=True)


X_train, scaler, pca = feature_reduction.to_PCA_comps(X_train, y_train)
X_test = feature_reduction.PCA_for_test(scaler, pca, X_test)


import aidalib.model_score_classification as model_score_classification

df_model_metric = model_score_classification.model_metrics(X_train, y_train)
print(df_model_metric)

df_search_best_param = model_score_classification.model_best_param(X_train, y_train)
print(df_search_best_param)
