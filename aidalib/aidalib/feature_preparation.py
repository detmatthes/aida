

def helper_split(df, target_columns, test_size=0.2, random_state=42):
    '''
    This function splits into train and test
    '''
    
    from sklearn.model_selection import train_test_split
    
    X = df.drop(columns=target_columns)
    y = df[target_columns]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    return X_train, X_test, y_train, y_test

"""
def empty_test(df, target_columns):
    X = df.drop(columns=target_columns)
    y = df[target_columns]
"""  