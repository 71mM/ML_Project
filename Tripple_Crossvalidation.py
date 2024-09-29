import numpy as np
from sklearn.metrics import classification_report, precision_score, make_scorer
from skopt import BayesSearchCV
from utils import save_model



def triple_cross_validation(model, X_train, X_val, X_test, y_train, y_val, y_test, param_grid, modelname):
    print("Tripple Cross Validierung:")
    precision_scorer = make_scorer(precision_score)
    model.fit(X_train, y_train)
    grid_search = BayesSearchCV(model, param_grid, cv=2, scoring=precision_scorer, verbose=3, n_jobs=-1)

    grid_search.fit(X_val, y_val)

    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_model.fit(np.vstack((X_train, X_val)), np.concatenate((y_train, y_val)))

    y_test_pred = best_model.predict(X_test)
    print(classification_report(y_test, y_test_pred))

    print("Beste Parameter:")
    for param in param_grid.keys():
        if param in best_params:
            print(f'{param}: {best_params[param]}')
    print("------------------------------------------------------------------------------------------------------------")
    save_model(best_model, modelname)
    return best_model
