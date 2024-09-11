import pandas as pd
import scipy.io
from sklearn.feature_extraction.text import TfidfTransformer


def load_data():
    path = f"Data/emails.mat"
    data = scipy.io.loadmat(path)

    X = data['X']
    X_dense = X.todense()
    Y = data['Y'].ravel()

    df_X = pd.DataFrame(X_dense)
    df_Y = pd.Series(Y, name='Spam')


    df_X = tf_idf(df_X)
    df_X = df_X.T
    return df_X, df_Y


def tf_idf(X):
    transformer = TfidfTransformer(smooth_idf=False)
    tfidf = transformer.fit_transform(X)
    tfidf = pd.DataFrame(tfidf.todense())
    return tfidf
