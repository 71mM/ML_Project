import numpy as np
from sklearn.metrics import classification_report, precision_score, make_scorer
from skopt import BayesSearchCV
from utils import save_model



def triple_cross_validation(model, X_train, X_val, X_test, y_train, y_val, y_test, param_grid, modelname):
    print("Tripple Cross Validierung:")
    no_spam_precision_scorer = make_scorer(precision_for_no_spam)
    model.fit(X_train, y_train)
    grid_search = BayesSearchCV(model, param_grid, cv=2,  scoring=no_spam_precision_scorer, n_jobs=-1, verbose=3)

    grid_search.fit(X_val, y_val)

    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_model.fit(np.vstack((X_train, X_val)), np.concatenate((y_train, y_val)))

    y_test_pred = best_model.predict(X_test)

    report = classification_report(y_test, y_test_pred, digits=4, output_dict=True)
    print(" - Precision für nicht Spam : ", report['-1']['precision'], "   - Precision für Spam: ", report['1']['precision'])
    print(" - Recall für nicht Spam :  ", report['-1']["recall"], "       - Recall für Spam:  ", report['1']["recall"])

    print("Beste Parameter:")
    for param in param_grid.keys():
        if param in best_params:
            print(f'{param}: {best_params[param]}')
    print("------------------------------------------------------------------------------------------------------------")
    save_model(best_model, modelname)
    return best_model


def precision_for_no_spam(y_true, y_pred):
    return precision_score(y_true, y_pred, labels=[-1], zero_division=0)
