import scipy.io
import pandas as pd
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import precision_recall_curve, auc, PrecisionRecallDisplay, classification_report, precision_score, confusion_matrix, recall_score
import matplotlib.pyplot as plt
import pickle
import os
import numpy as np
import seaborn as sns


def load_data():
    """
    Lädt die Daten aus einer .mat-Datei und bereitet sie für die Verwendung vor.

    :return: Zwei DataFrames: df_X enthalten die Merkmale und df_Y enthalten die Labels.
    """
    path = "Data/emails.mat"
    data = scipy.io.loadmat(path)

    X = data['X']
    X_dense = X.todense()
    Y = data['Y'].ravel()

    df_X = pd.DataFrame(X_dense)
    df_Y = pd.Series(Y, name='Spam')

    df_X = df_X.T

    return df_X, df_Y


def tf_idf(X):
    """
    Berechnet den TF-IDF-Wert (Term Frequency-Inverse Document Frequency) für die gegebenen Daten.

    :param X: DataFrame, der die Eingabedaten enthält. Die Spaltennamen repräsentieren Merkmale, und die Zeilen repräsentieren Dokumente.
    :return: DataFrame mit den TF-IDF-werten, wobei die Spaltennamen beibehalten werden.
    """
    X.columns = X.columns.astype(str)
    transformer = TfidfTransformer(smooth_idf=True, use_idf=True)
    tfidf = transformer.fit_transform(X)
    tfidf_df = pd.DataFrame(tfidf.todense(), columns=X.columns)

    return tfidf_df



def determine_better_data(model1, name1, model2, name2, X1, X2, y, dummy, dummy_name="Baseline Model"):
    """
    Vergleicht zwei Modelle basierend auf deren Precision-Recall-Kurven und gibt das bessere Modell zurück.

    :param model1: Erstes Modell zur Bewertung.
    :param name1: Name des ersten Modells.
    :param model2: Zweites Modell zur Bewertung.
    :param name2: Name des zweiten Modells.
    :param X1: Merkmale der Daten für das erste Modell.
    :param X2: Merkmale der Daten für das zweite Modell.
    :param y: Wahre Labels.
    :param dummy: Baseline-Modell zum Vergleich.
    :param dummy_name: Name des Baseline-Modells. Standard ist "Baseline Model".
    :return: Ein Wörterbuch mit Informationen über das bessere Modell und dessen Leistungskennzahlen (Precision, Recall, AUC).
    """

    if model1 is not None and hasattr(model1, 'predict_proba'):
        y_score_1 = model1.predict_proba(X1)[:, 1]
        y_score_2 = model2.predict_proba(X2)[:, 1]
    elif model1 is not None and hasattr(model1, 'decision_function'):
        y_score_1 = model1.decision_function(X1)
        y_score_2 = model2.decision_function(X2)

    y_score_dummy = dummy.predict_proba(X2)[:, 1]
    if model1 is not None:
        precision_1, recall_1, _ = precision_recall_curve(y, y_score_1)
        precision_2, recall_2, _ = precision_recall_curve(y, y_score_2)
    precision_dummy, recall_dummy, _ = precision_recall_curve(y, y_score_dummy)

    if model1 is not None:
        auc_1 = auc(recall_1, precision_1)
        auc_2 = auc(recall_2, precision_2)
    auc_dummy = auc(recall_dummy, precision_dummy)

    fig, ax = plt.subplots()
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall-Kurve zum Modellvergleich')

    if model1 is not None:
        display_1 = PrecisionRecallDisplay(precision=precision_1, recall=recall_1)
        display_1.plot(ax=ax, label=name1 + f" (AUC = {auc_1:.2f})")

        display_2 = PrecisionRecallDisplay(precision=precision_2, recall=recall_2)
        display_2.plot(ax=ax, label=name2 + f" (AUC = {auc_2:.2f})")

    display_dummy = PrecisionRecallDisplay(precision=precision_dummy, recall=recall_dummy)
    display_dummy.plot(ax=ax, label=dummy_name + f" (AUC = {auc_dummy:.2f})")

    ax.legend(loc='lower right')

    save_folder = 'Data/bilder'
    plt.savefig(os.path.join(save_folder, f'{name1}_visualization.pdf'))
    plt.show()

    if model1 is not None:
        y_score_no_spam_1 = model1.predict(X1)
        y_score_no_spam_2 = model2.predict(X2)
        report_1 = classification_report(y, y_score_no_spam_1, output_dict=True)
        report_2 = classification_report(y, y_score_no_spam_2, output_dict=True)
    else:
        better_model = dummy
        better_data = "Baseline Model"
        better_precision = precision_dummy
        better_recall = recall_dummy
        better_auc = auc_dummy
        print("Das beste Modell ist das Baseline-Modell.")
        return {
            'better_data': better_data,
            'better_model': better_model,
            'better_precision': better_precision,
            'better_recall': better_recall,
            'better_auc': better_auc
        }

    precision_dummy_no_spam = precision_score(y, y_score_dummy, zero_division=0)

    if report_1["-1"]['precision'] >= report_2["-1"]['precision'] and report_1["-1"][
        'precision'] > precision_dummy_no_spam:
        better_model = model1
        better_data = name1
        better_precision = [report_1["-1"]['precision'], report_1["1"]['precision']]
        better_recall = [report_1["-1"]['recall'], report_1["1"]['recall']]
        better_auc = auc_1
    elif report_2["-1"]['precision'] > report_1["-1"]['precision'] and report_2["-1"][
        'precision'] > precision_dummy_no_spam:
        better_model = model2
        better_data = name2
        better_precision = [report_2["-1"]['precision'], report_2["1"]['precision']]
        better_recall = [report_2["-1"]['recall'], report_2["1"]['recall']]
        better_auc = auc_2
    else:
        better_model = dummy
        better_data = "Baseline Model"
        better_precision = precision_dummy
        better_recall = recall_dummy
        better_auc = auc_dummy
        print("Das beste Modell ist das Baseline-Modell.")

    return {
        'better_data': better_data,
        'better_model': better_model,
        'better_precision': better_precision,
        'better_recall': better_recall,
        'better_auc': better_auc
    }


def find_best_model(*models):
    """
    Findet das beste Modell basierend auf mehreren Metriken und erstellt Visualisierungen.

    :param models: Eine beliebige Anzahl von Modellen zusammen mit ihren Testdaten, gegeben als Dictionaries,
                   die folgende Schlüssel enthalten:
                   - 'name': Name des Modells
                   - 'model': Das Modellobjekt
                   - 'X_test': Testdatenmerkmale
                   - 'y_test': Testdatenlabels
    :return: Ein Wörterbuch mit Rankings der Modelle basierend auf Precision, Recall und AUC.
    """
    results = []

    for model_entry in models:
        model_name = model_entry['name']
        model = model_entry['model']
        X_test = model_entry['X_test']
        y_test = model_entry['y_test']

        y_pred = model.predict(X_test)
        y_score = np.where(y_pred == -1, -1, 1)

        precision, recall, _ = precision_recall_curve(y_test, y_score)
        model_auc = auc(recall, precision)

        fig, axs = plt.subplots(1, 2, figsize=(15, 6))


        axs[0].plot(recall, precision, label=f'{model_name} (AUC={model_auc:.2f})')
        axs[0].set_xlabel('Recall')
        axs[0].set_ylabel('Precision')
        axs[0].set_title(f'Precision-Recall Curve: {model_name}')
        axs[0].legend()


        cm = confusion_matrix(y_test, y_pred, labels=[-1, 1])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axs[1],
                    xticklabels=['nicht Spam', 'Spam'],
                    yticklabels=['nicht Spam', 'Spam'])
        axs[1].set_xlabel('Vorhergesagt')
        axs[1].set_ylabel('Tatsächlich')
        axs[1].set_title(f'Confusion Matrix: {model_name}')

        plt.tight_layout()
        save_folder = 'Data/bilder'
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        plt.savefig(os.path.join(save_folder, f'visualization_{model_name}.pdf'))
        plt.show()

        report = classification_report(y_test, y_pred, digits=4, output_dict=True)
        results.append({
            'model_name': model_name,
            'precision': [report['-1']['precision'], report['1']['precision']],
            'recall': [report['-1']["recall"], report['1']["recall"]],
            'auc': model_auc,
            'model': model
        })

    precision_ranking = sorted(results, key=lambda x: x['precision'][0], reverse=True)
    recall_ranking = sorted(results, key=lambda x: x['recall'][0], reverse=True)
    overall_ranking = sorted(results, key=lambda x: x['auc'], reverse=True)

    return {
        'precision_ranking': precision_ranking,
        'recall_ranking': recall_ranking,
        'overall_ranking': overall_ranking
    }



def save_model(model, modelname):
    """
    Speichert ein gegebenes Modell in einer Datei.

    :param model: Das zu speichernde Modell.
    :param modelname: Der Name, unter dem das Modell gespeichert werden soll.
    """
    directory = 'Data/modelle'
    os.makedirs(directory, exist_ok=True)
    filename = os.path.join(directory, modelname)

    with open(filename, 'wb') as file:
        pickle.dump(model, file)

    print(f"Modell gespeichert als {filename}")
