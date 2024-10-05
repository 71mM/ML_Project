from skopt.space import Real, Integer, Categorical

# Parameter Grid für Decision Tree
decision_tree_grid = {
    'criterion': Categorical(['gini', 'entropy']),
    'max_depth': Categorical([None] + list(range(1, 111, 10))),
    'min_samples_split': Integer(2, 10),
    'min_samples_leaf': Integer(1, 4),
    'max_features': Categorical([None, 'sqrt', 'log2']),
    'min_impurity_decrease': Real(0.0, 0.1)  # Optional
}


# Parameter Grid für Random Forest
random_forest_grid = {
    'n_estimators': Integer(50, 200),
    'max_depth': Categorical([None] + list(range(1, 111, 10))),
    'min_samples_split': Integer(2, 10),
    'min_samples_leaf': Integer(1, 4),
    'bootstrap': Categorical([True, False]),
    'max_features': Categorical(['sqrt', 'log2']),
}

# Parameter Grid für Logistische Regression
logistische_regression_grid = {
    'penalty': Categorical(['l1', 'l2']),
    'C': Real(0.01, 1, prior='log-uniform'),
    'solver': Categorical(['liblinear', 'saga']),
    'max_iter': Integer(500, 1000),
    'tol': Real(1e-4, 1e-2, prior='log-uniform'),
    'class_weight': Categorical(['balanced'])  # Optional
}

# Parameter Grid für Lineare Klassifikation
lineare_klassifikation_grid = {
    'loss': Categorical(['hinge', 'squared_hinge']),  # Keep this if you're using a model that supports it
    'C': Real(0.0001, 1, prior='log-uniform'),
    'max_iter': Integer(1000, 5000),
    'penalty': Categorical(['l2']),  # Note: Check if this is compatible with your model
    'tol': Real(1e-4, 1e-2, prior='log-uniform'),
}

# Parameter für SVM
svm_grid = {
    'C': Real(0.01, 1, prior='log-uniform'),
    'gamma': Categorical(['scale', 'auto']),
    'tol': Real(1e-4, 1e-2, prior='log-uniform'),
    'max_iter': Integer(1000, 5000)
}