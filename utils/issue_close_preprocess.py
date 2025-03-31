import numpy as np

def preprocess(X_train, y_train, X_test, y_test,  filename, classes = 2):
    """preprocess y values and x values for both training and test data"""
    y_train = __preprocess_target__(y_train, classes, filename)
    if y_test:
        y_test = __preprocess_target__(y_test, classes, filename)
    X_train = __preprocess_features__(X_train, filename)
    if X_test:
        X_test = __preprocess_features__(X_test,filename)
    return X_train, y_train, X_test, y_test

def __preprocess_target__(y, n_classes, filename):
    if n_classes == 2:
        #reversed
        if 'firefox.csv' in filename:
            y = np.where(y < 4, 0, 1)
        elif 'chromium.csv' in filename:
            y = np.where(y < 5, 0, 1)
        else:
            y = np.where(y < 6, 0, 1)
    elif n_classes == 3:
        y = np.where(y < 2, 0,
                                np.where(y < 6, 1, 2))
    elif n_classes == 5:
        y = np.where(y < 1, 0, np.where(y < 3, 1, np.where(
            y < 6, 2, np.where(y < 21, 3, 4))))
    elif n_classes == 7:
        y = np.where(y < 1, 0,
                                np.where(y < 2, 1, np.where(y < 3, 2, np.where(
                                    y < 6, 3,
                                    np.where(y < 11, 4, np.where(y < 21, 5, 6))))))
       
    else:
        y = np.where(y < 1, 0, np.where(y < 2, 1, np.where(y < 3, 2, np.where(y < 4, 3,
                                                                              np.where(y < 6, 4, np.where(y < 8,5, 
                                                                                                          np.where( y < 11, 6, np.where(y < 21, 7, 8))))))))
    return y

def __preprocess_features__(X, filename = None):
    """Remove name, version name and package from features"""
    X.drop(['Unnamed: 0', 'bugID'], axis=1, inplace=True)
    X1 = X[['s1', 's2', 's3', 's4', 's5', 's6', 's8']].copy()
    X1['s70'] = X['s7'].apply(lambda x: eval(x)[0])
    X1['s71'] = X['s7'].apply(lambda x: eval(x)[1])
    X1['s72'] = X['s7'].apply(lambda x: eval(x)[2])
    X1['s90'] = X['s9'].apply(lambda x: eval(x)[0])
    X1['s91'] = X['s9'].apply(lambda x: eval(x)[1])

    if 'firefox' in filename:
        X1['s92'] = X['s9'].apply(lambda x: eval(x)[2])
    return X1