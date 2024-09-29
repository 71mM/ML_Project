from skopt.space import Real, Integer, Categorical

# Parameter Grid für Decision Tree
decision_tree_grid = {
    'criterion': Categorical(['gini', 'entropy']),
    'max_depth': Categorical([None] + list(range(10, 41, 10))),
    'min_samples_split': Integer(2, 10),
    'min_samples_leaf': Integer(1, 4),
}

# Parameter Grid für Random Forest
random_forest_grid = {
    'n_estimators': Integer(50, 200),
    'max_depth': Categorical([None] + list(range(10, 41, 10))),
    'min_samples_split': Integer(2, 10),
    'min_samples_leaf': Integer(1, 4),
    'bootstrap': Categorical([True, False]),
}

# Parameter Grid für Logistische Regression
logistische_regression_grid = {
    'penalty': Categorical(['l1', 'l2', 'elasticnet']),
    'C': Real(0.01, 10, prior='log-uniform'),
    'solver': Categorical(['liblinear', 'saga']),
    'max_iter': Integer(100, 500),
}

# Parameter Grid für Lineare Klassifikation (SGDClassifier)
lineare_klassifikation_grid = {
    'loss': Categorical(['hinge', 'log']),
    'alpha': Real(0.0001, 0.01, prior='log-uniform'),
    'max_iter': Integer(1000, 5000),
    'penalty': Categorical(['l2', 'l1', 'elasticnet']),
}

# Parameter Grid für SVM (Support Vector Machine) als Ersatz für Lineare Regression
svm_grid = {
    'C': Real(0.01, 10, prior='log-uniform'),
    'kernel': Categorical(['linear', 'rbf']),
    'gamma': Categorical(['scale', 'auto']),
}