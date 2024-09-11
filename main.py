from utils import load_data
from rforest import random_forest
from svm import svm
from lreg import logistic_regression
from sklearn.model_selection import train_test_split

X, y = load_data()

test_size = 0.2
val_size = 0.2
random_state = 42
n_splits = 10

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=(test_size + val_size), random_state=random_state)

test_val_split = val_size / (test_size + val_size)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=test_val_split, random_state=random_state)

#random_forest(X_train, X_test, y_train, y_test)
svm(X_train, X_test, X_val, y_train, y_test, y_val, random_state, n_splits)
#logistic_regression(X_train, X_test, y_train, y_test)
