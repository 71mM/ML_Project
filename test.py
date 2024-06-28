import pandas as pd
import scipy.io

path = f"Data/emails.mat"

data = scipy.io.loadmat(path)

X = data['X']
X_dense = X.todense()
Y = data['Y']

df_X = pd.DataFrame(X_dense)
df_Y = pd.DataFrame(Y)


df_X = pd.concat([df_X, df_Y], axis=1)
df_X = df_X.transpose()

df_X.to_csv('Data/emails.csv', index=False, header=False)
