import numpy as np
from sklearn.metrics import classification_report, precision_score, make_scorer
from skopt import BayesSearchCV
from utils import save_model
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
import os



def triple_cross_validation(model, X_train, X_val, X_test, y_train, y_val, y_test, param_grid, modelname):
    """
    Führt eine dreifache Kreuzvalidierung mit BayesSearch auf dem Modell durch und plottet die Lernkurve.

    :param model: Das zu trainierende Modell.
    :param X_train: Die Merkmale der Trainingsdaten.
    :param X_val: Die Merkmale der Validierungsdaten.
    :param X_test: Die Merkmale der Testdaten.
    :param y_train: Die Labels der Trainingsdaten.
    :param y_val: Die Labels der Validierungsdaten.
    :param y_test: Die Labels der Testdaten.
    :param param_grid: Der Parameterraum für die BayesSearchCV.
    :param modelname: Der Name unter dem das trainierte Modell gespeichert wird.
    :return: Das beste trainierte Modell nach BayesSearchCV.
    """
    print("Triple Cross Validierung:")
    no_spam_precision_scorer = make_scorer(precision_for_no_spam)
    bayes_search = BayesSearchCV(model, param_grid, cv=2, scoring=no_spam_precision_scorer, n_jobs=-1, verbose=3)

    bayes_search.fit(X_val, y_val)

    best_model = bayes_search.best_estimator_
    best_params = bayes_search.best_params_
    best_model.fit(np.vstack((X_train, X_val)), np.concatenate((y_train, y_val)))

    y_test_pred = best_model.predict(X_test)
    plot_learning_curve(best_model, modelname, np.vstack((X_train, X_val)), np.concatenate((y_train, y_val)), cv=2,
                        n_jobs=-1)

    report = classification_report(y_test, y_test_pred, digits=4, output_dict=True)
    print(" - Genauigkeit für nicht Spam: ", report['-1']['precision'], "   - Genauigkeit für Spam: ",
          report['1']['precision'])
    print(" - Abrufrate für nicht Spam:  ", report['-1']["recall"], "       - Abrufrate für Spam:  ",
          report['1']["recall"])

    print("Beste Parameter:")
    for param in param_grid.keys():
        if param in best_params:
            print(f'{param}: {best_params[param]}')
    print("----------------------------------------------------------------------------------------------------------")
    save_model(best_model, modelname)
    return best_model


def precision_for_no_spam(y_true, y_pred):
    """
    Berechnet die Genauigkeit (Precision) für nicht Spam Labels.

    :param y_true: Die wahren Labels.
    :param y_pred: Die vorhergesagten Labels.
    :return: Die Genauigkeit für nicht Spam Labels.
    """
    return precision_score(y_true, y_pred, labels=[-1], zero_division=0)


def plot_learning_curve(estimator, title, X, y, cv=None, n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Plottet die Lernkurve für einen gegebenen Schätzer.

    :param estimator: Das zu bewertende Modell.
    :param title: Der Titel des Plots.
    :param X: Die Merkmale der Daten.
    :param y: Die Labels, die den Datenmerkmalen entsprechen.
    :param cv: Die Cross-Validation-Strategie. Standard ist None.
    :param n_jobs: Anzahl der parallel auszuführenden Jobs. Standard ist 1.
    :param train_sizes: Die verschiedenen Größen der Trainingsdaten als Brüche eines Ganzzahlbereichs. Standard ist np.linspace(.1, 1.0, 5).
    :return: Das erstellte Plot-Objekt.
    """
    plt.figure()
    plt.title(title)
    plt.xlabel("Anzahl der Trainingsbeispiele")
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
             label="Trainings-Score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-Validation-Score")

    plt.legend(loc="best")
    save_folder = 'Data/bilder'
    plt.savefig(os.path.join(save_folder, f'{title}_visualization.pdf'))
    plt.show()

    return plt
