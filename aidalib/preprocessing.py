import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder


def list_possible_categorical_features(df):
    """
    returns names of possible categorical features
    """
    return list(df.select_dtypes(include=['category','object']).columns)

def check_feature_lists(possible_categorical_features, discrete_categorical_features, ordinal_categorical_features):
    possible_categorical_features_set = set(possible_categorical_features)
    discrete_categorical_features_set = set(discrete_categorical_features)
    ordinal_categorical_features_set = set(ordinal_categorical_features)
    if not discrete_categorical_features_set.issubset(possible_categorical_features_set):
        unknown_features = str(discrete_categorical_features_set-possible_categorical_features_set)
        print("unknown discrete features" + unknown_features)
        raise("discrete list contains unknown feature names")
    if not ordinal_categorical_features_set.issubset(possible_categorical_features_set):
        unknown_features = str(ordinal_categorical_features_set-possible_categorical_features_set)
        print("unknown ordinal features" + unknown_features)
        raise("ordinal list contains unknown feature names")
    if not ordinal_categorical_features_set.isdisjoint(discrete_categorical_features_set):
        common_features = str(ordinal_categorical_features_set.intersection(discrete_categorical_features_set))
        print("common features" + common_features)
        raise("ordinal list and discrete list contains common feature names")
    missing_features = possible_categorical_features_set - discrete_categorical_features_set - ordinal_categorical_features_set
    if len(missing_features) > 0:
        print("WARN: mission possible features", missing_features)

def fill_na_one_categorical_features(df, categorical_features):
    df_filled = df.copy()
    for col in categorical_features:
        df_filled[col] = df[col].fillna('default')
    return df_filled

def encode_discrete_categorical_features(df, discrete_categorical_features, oh_encoder=None):
    if oh_encoder == None:
        oh_encoder = OneHotEncoder(sparse=False).fit(df[discrete_categorical_features])
    df_ohe = pd.DataFrame(oh_encoder.transform(df[discrete_categorical_features]), columns=oh_encoder.get_feature_names(discrete_categorical_features))
    df_ohe.index = df.index
    df_with_ohe = pd.concat([df, df_ohe], axis=1).drop(columns=discrete_categorical_features)
    return df_with_ohe, oh_encoder

def encode_ordinal_categorical_features(df, ordinal_categorical_features, ord_encoder=None):
    df_with_enc = df.copy()
    if ord_encoder == None:
        ord_encoder = OrdinalEncoder().fit(df[ordinal_categorical_features])
    df_with_enc[ordinal_categorical_features] = ord_encoder.transform(df[ordinal_categorical_features])
    return df_with_enc, ord_encoder

