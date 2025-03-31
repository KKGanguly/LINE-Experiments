def preprocess(X, y):
    """preprocess y values and x values for both training and test data"""
    X = __preprocess_features__(X)
    y = __preprocess_target__(y, X.index)
    return X, y

def __preprocess_target__(y, select):
    """Convert the target column into strings '0' and '1'."""
    y = y.loc[select]    
    return y

def __preprocess_features__(X):
    """Remove duplicates from features"""
    X = X.drop_duplicates()
    return X

