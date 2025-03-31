from utils.CFS import CFS


def preprocess(X, y):
    """preprocess y values and x values for both training and test data"""
    X = __preprocess_features__(X)
    y = __preprocess_target__(y, X.index)

    return X, y

def __preprocess_target__(y, select):
    """Convert the target column into strings '0' and '1'."""
    y = y.apply(lambda x: 0 if x == 0 else 1)
    y = y.loc[select]
    return y

def __preprocess_features__(X):
    """Remove name, version name and package from features"""
    X.drop(['name','version','name.1'], errors='ignore', axis =1, inplace= True)
    X = X.drop_duplicates()
    return X

