import pandas as pd
import scipy.io


def create_df():
    path = f"Data/emails.mat"
    data = scipy.io.loadmat(path)

    X = data['X']
    X_dense = X.todense()
    Y = data['Y']

    df_X = pd.DataFrame(X_dense)
    df_Y = pd.DataFrame(Y)

    df_X = df_X.T
    df_Y = df_Y.T

    df_Y = df_Y.rename(columns={0: 'Spam'})
    df_Y['Spam'] = df_Y['Spam'].replace(1, 'Yes')
    df_Y['Spam'] = df_Y['Spam'].replace(-1, 'No')

    df = pd.concat([df_X, df_Y], axis=1)
    return df
