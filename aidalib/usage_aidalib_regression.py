import warnings
warnings.filterwarnings("ignore")

import pandas as pd

# example: Projekt "House prices"


df = pd.read_csv('raw_data.csv')
df.drop(columns='Train', inplace=True)
targetColumn = 'SalePrice'

df[targetColumn].isna()

# remove entries with NaN target
df = df[~df[targetColumn].isna()]


# PreProcessing

df['LotFrontage'].hist()
df['LotFrontage'].mean()
df['LotFrontage'].fillna(df['LotFrontage'].mean(), inplace=True)

df['MasVnrArea'].hist()
df['MasVnrArea'].median()
df['MasVnrArea'].fillna(df['MasVnrArea'].median(), inplace=True)

df['GarageYrBlt'].hist()
df['GarageYrBlt'].mean()
df['GarageYrBlt'].fillna(df['GarageYrBlt'].median(), inplace=True)



import aidalib.preprocessing as preprocessing

possible_categorical_features = preprocessing.list_possible_categorical_features(df)
#print("Possible categorical features", possible_categorical_features)

# differentiation between discrete and ordinal categorical features
discrete_categorical_features = ['Exterior1st','BsmtExposure','SaleCondition','SaleType','Condition1','Exterior2nd','Condition2','RoofStyle','MSZoning', 'Street', 'Alley', 'Utilities', 'LotConfig', 'Neighborhood', 'BldgType', 'HouseStyle', 'RoofMatl', 'MasVnrType', 'Foundation', 'Heating', 'Electrical', 'GarageType', 'MiscFeature']
ordinal_categorical_features = ['GarageCond','BsmtFinType2','GarageFinish','LotShape', 'LandContour', 'LandSlope', 'ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'BsmtFinType1', 'HeatingQC', 'CentralAir', 'KitchenQual', 'Functional', 'FireplaceQu', 'GarageQual', 'PavedDrive', 'PoolQC', 'Fence']
preprocessing.check_feature_lists(possible_categorical_features, discrete_categorical_features, ordinal_categorical_features)


# feature_preparation: helper_split
import aidalib.feature_preparation as feature_preparation

X_train, X_test, y_train, y_test = feature_preparation.helper_split(df, targetColumn, test_size=0.2, random_state=99)


# encode categorical features

X_train = preprocessing.fill_na_one_categorical_features(X_train, possible_categorical_features)
X_test = preprocessing.fill_na_one_categorical_features(X_test, possible_categorical_features)

X_train, oh_encoder = preprocessing.encode_discrete_categorical_features(X_train, discrete_categorical_features)
X_test, _ = preprocessing.encode_discrete_categorical_features(X_test, discrete_categorical_features, oh_encoder)

X_train, ord_encoder = preprocessing.encode_ordinal_categorical_features(X_train, ordinal_categorical_features)
X_test, _ = preprocessing.encode_ordinal_categorical_features(X_test, ordinal_categorical_features, ord_encoder)


# TO functions for null values





# reduce features

import aidalib.feature_reduction as feature_reduction

columns_with_low_correlated_features_to_target, _ = feature_reduction.determine_low_correlated_features_to_target(X_train, y_train)
X_train.drop(columns=columns_with_low_correlated_features_to_target, inplace=True)
X_test.drop(columns=columns_with_low_correlated_features_to_target, inplace=True)

columns_with_strong_correlation_between, _ = feature_reduction.determine_strong_correlated_features(X_train, y_train)
X_train.drop(columns=columns_with_strong_correlation_between, inplace=True)
X_test.drop(columns=columns_with_strong_correlation_between, inplace=True)

df_feature_importances = feature_reduction.determine_feature_importance(X_train, y_train)
X_train = feature_reduction.drop_unimportant_features(X_train, df_feature_importances)
X_test = feature_reduction.drop_unimportant_features(X_test, df_feature_importances)

X_train, scaler, pca = feature_reduction.to_PCA_comps(X_train, y_train)
X_test = feature_reduction.PCA_for_test(scaler, pca, X_test)



import aidalib.model_score_regression as model_score_regression


df_model_metric = model_score_regression.init_df_model_metric_regression()
model_score_regression.model_metric_LinearRegression(X_train, y_train, df_model_metric)
model_score_regression.model_metric_DecisionTreeRegressor(X_train, y_train, df_model_metric)
model_score_regression.model_metric_RandomForestRegressor(X_train, y_train, df_model_metric)
model_score_regression.model_metric_AdaBoostRegressor(X_train, y_train, df_model_metric)
model_score_regression.model_metric_GradientBoostingRegressor(X_train, y_train, df_model_metric)
print(df_model_metric)

#regression_models = model_score_regression.getRegressionModels()
#for regression_model in regression_models:
#    model_score_regression.model_metric_regression(regression_model, X_train, y_train, df_model_metric)

df_search_best_param = model_score_regression.init_df_search_best_param()
model_score_regression.search_bestparam_LinearRegression(X_train, y_train, df_search_best_param)
model_score_regression.search_bestparam_DecisionTreeRegressor(X_train, y_train, df_search_best_param)
model_score_regression.search_bestparam_RandomForestRegressor(X_train, y_train, df_search_best_param)
print(df_search_best_param)
