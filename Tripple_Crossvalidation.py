import numpy as np
from sklearn.metrics import classification_report, precision_score, make_scorer
from skopt import BayesSearchCV
from utils import save_model
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
import os



def triple_cross_validation(model, X_train, X_val, X_test, y_train, y_val, y_test, param_grid, modelname):
    return model
    print("Tripple Cross Validierung:")
    no_spam_precision_scorer = make_scorer(precision_for_no_spam)
    bayes_search = BayesSearchCV(model, param_grid, cv=2,  scoring=no_spam_precision_scorer, n_jobs=-1, verbose=3)

    bayes_search.fit(X_val, y_val)

    best_model = bayes_search.best_estimator_
    best_params = bayes_search.best_params_
    best_model.fit(np.vstack((X_train, X_val)), np.concatenate((y_train, y_val)))

    y_test_pred = best_model.predict(X_test)
    plot_learning_curve(best_model, modelname, np.vstack((X_train, X_val)), np.concatenate((y_train, y_val)), cv=2, n_jobs=-1)
    report = classification_report(y_test, y_test_pred, digits=4, output_dict=True)
    print(" - Precision für nicht Spam : ", report['-1']['precision'], "   - Precision für Spam: ", report['1']['precision'])
    print(" - Recall für nicht Spam :  ", report['-1']["recall"], "       - Recall für Spam:  ", report['1']["recall"])

    print("Beste Parameter:")
    for param in param_grid.keys():
        if param in best_params:
            print(f'{param}: {best_params[param]}')
    print("----------------------------------------------------------------------------------------------------------")
    save_model(best_model, modelname)
    return best_model


def precision_for_no_spam(y_true, y_pred):
    return precision_score(y_true, y_pred, labels=[-1], zero_division=0)


def plot_learning_curve(estimator, title, X, y, cv=None, n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure()
    plt.title(title)
    plt.xlabel("Training examples")
    plt.ylabel("Score")

    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.grid()
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1,
                     color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    plt.show()
    save_folder = 'Data/bilder'
    plt.savefig(os.path.join(save_folder, f'{title}_visualization.pdf'))
    return plt
