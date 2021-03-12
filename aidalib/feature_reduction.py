import pandas as pd
from sklearn.preprocessing import StandardScaler
import sklearn.decomposition as decomposition
from sklearn.tree import DecisionTreeClassifier


def determine_low_correlated_features_to_target(X, y):
    df_corr_to_target = pd.DataFrame(columns=['abs_corr_to_target','abs_corr_to_target_rel'], dtype='float')
    #corr = pd.concat([X, y],axis=1).phik_matrix()
    corr = pd.concat([X, y],axis=1).corr()
    corr[y.name] = abs(corr[y.name])
    #print(corr)
    sum_target_corr = sum(corr[y.name])-1
    j = corr.shape[0]
    for i in range(corr.shape[0]-1):
        col_name = corr.index[i]
        corr_to_target = corr.iloc[i,j-1]
        corr_to_target_rel = corr_to_target / sum_target_corr
        df_corr_to_target.loc[col_name] = [corr_to_target, corr_to_target_rel]
    # TODO verwende df_corr_to_target.descripe() den Wert unter 25%
    min_abs_corr = df_corr_to_target.describe().iloc[4,0]
    columns_with_low_correlated_features_to_target = list(df_corr_to_target[df_corr_to_target['abs_corr_to_target'] <= min_abs_corr].index)
    print(f"Found {len(columns_with_low_correlated_features_to_target)} of {X.shape[1]} columns with low correlation to target :{columns_with_low_correlated_features_to_target}")
    return columns_with_low_correlated_features_to_target, df_corr_to_target.sort_values('abs_corr_to_target')



def determine_strong_correlated_features(X, y):
    corr_to_target = pd.concat([X, y], axis=1).corr().iloc[:-1,-1:]
    df_corr_between_columns = pd.DataFrame(columns=['correlation','col1','col2', 'corr1ToTarget', 'corr2ToTarget', 'colWithLowestCorrToTarget'])
    corr = X.corr()
    i = 0
    for i in range(corr.shape[0]):
        for j in range(i+1, corr.shape[0]):
            col1 = corr.index[i]
            col2 = corr.columns[j]
            corr1ToTarget = abs(corr_to_target.loc[col1].values[0])
            corr2ToTarget = abs(corr_to_target.loc[col2].values[0])
            colWithLowestCorrToTarget = col1
            if corr1ToTarget > corr2ToTarget:
                colWithLowestCorrToTarget = col2
            corr_val = abs(corr.iloc[i,j])
            if col1 != col2:
                df_corr_between_columns.at[i] = [corr_val, col1, col2, corr1ToTarget, corr2ToTarget, colWithLowestCorrToTarget]
                i = i + 1
                #if corr_val >= 0.9:
                #    print(f"Strong corr {corr_val:.2f} between X-columns '{col1}' and '{col2}'")
    #  TODO corr zu target mit beachten
    df_corr_between_columns.sort_values('correlation', ascending=False, inplace=True)
    columns_with_strong_correlation_between = df_corr_between_columns[df_corr_between_columns['correlation'] >= 0.72]['colWithLowestCorrToTarget'].values
    print(f"Found {len(columns_with_strong_correlation_between)} of {X.shape[1]} columns with strong correlation in between :{columns_with_strong_correlation_between}")
    return columns_with_strong_correlation_between, df_corr_between_columns


def determine_feature_importance(X, y):
    modelDTC = DecisionTreeClassifier()
    modelDTC.fit(X, y)
    df_feature_importances = pd.DataFrame(modelDTC.feature_importances_, index = X.columns, columns=['feature importance'])
    df_feature_importances = df_feature_importances.sort_values('feature importance')
    num_unimportant_features = (df_feature_importances['feature importance'] == 0).sum()
    num_features = len(df_feature_importances)
    print(f"determine_feature_importance: found {num_unimportant_features} unimportant features of {num_features}")
    del modelDTC
    return df_feature_importances


def drop_unimportant_features(X, df_feature_importances, min_importance = 0) -> pd.DataFrame:
    unimportant_features = list(df_feature_importances[df_feature_importances['feature importance'] <= min_importance].index)
    print(f"drop_unimportant_features: drop unimportant features: {unimportant_features}")
    X_d = X.copy()
    X_d.drop(columns=unimportant_features, inplace=True)
    return X_d


def to_PCA_comps(X, y, n_components=0.95):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = decomposition.PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    print(f"to_PCA_comps {X.shape[1]} -> {X_pca.shape[1]}")
    columns = ['pca_%i' % i for i in range(pca.n_components_)]
    df_pca = pd.DataFrame(X_pca, columns=columns, index=X.index)
    return df_pca, scaler, pca

def PCA_for_test(scaler, pca, X_test):
    X_scaled = scaler.transform(X_test)
    X_pca = pca.transform(X_scaled)
    print(f"PCA_for_test {X_test.shape[1]} -> {X_pca.shape[1]}")
    columns = ['pca_%i' % i for i in range(pca.n_components_)]
    df_pca_test = pd.DataFrame(X_pca, columns=columns, index=X_test.index)
    return df_pca_test
