import pandas as pd
import scipy.io


def load_data():
    path = f"Data/emails.mat"
    data = scipy.io.loadmat(path)

    X = data['X']
    X_dense = X.todense()
    Y = data['Y'].ravel()

    df_X = pd.DataFrame(X_dense)
    df_Y = pd.Series(Y, name='Spam')

    df_X = df_X.T

    return df_X, df_Y
