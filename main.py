from utils import load_data
from rforest import random_forest
from svm import svm
from lreg import logistic_regression
from sklearn.model_selection import train_test_split

X, y = load_data()
print(y.value_counts())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

random_forest(X_train, X_test, y_train, y_test)
svm(X_train, X_test, y_train, y_test)
logistic_regression(X_train, X_test, y_train, y_test)
