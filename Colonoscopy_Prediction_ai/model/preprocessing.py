import numpy as np

def shuffle_data(df):
    names = df.iloc[:,0].values
    df = df.iloc[:,1:].astype('float64')

    X = df.iloc[:,1:].values
    y = df.iloc[:,0].values.reshape(-1,1)

    indices = []

    unique_names = np.unique(names)

    np.random.seed(7)
    np.random.shuffle(unique_names)

    for name in unique_names:
        temp = np.where(names == name)[0]
        indices.extend(temp)

    X = X[indices]
    y = y[indices]

    return X,y

def split_data(df):

    X,y = shuffle_data(df)

    X_test = X[-20:,:]
    y_test = y[-20:,:]

    X_train = X[:-20,:]
    y_train = y[:-20,:]

    return X_test,y_test,X_train,y_train

def normalization(X_train):

    X_train_mean = np.mean(X_train,axis=0,keepdims=True)
    X_train_std = np.std(X_train,axis=0,keepdims=True)
    X_train_std[X_train_std == 0] = 1

    X_train_norm = (X_train - X_train_mean) /  X_train_std

    return X_train_norm,X_train_mean,X_train_std

def hot_encoding(y,n_classes):
    m = y.shape[0]

    one_hot = np.zeros((m,n_classes))

    for p in range(m):
        one_hot[p,int(y[p,0])-1] = 1

    return one_hot