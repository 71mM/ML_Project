import scipy.io
import pandas as pd
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import precision_recall_curve, auc, PrecisionRecallDisplay
import matplotlib.pyplot as plt
import pickle
import os


def load_data():
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
    X.columns = X.columns.astype(str)  # Konvertiere Spaltennamen in Strings
    transformer = TfidfTransformer(smooth_idf=True, use_idf=True)
    tfidf = transformer.fit_transform(X)
    tfidf_df = pd.DataFrame(tfidf.todense(), columns=X.columns)  # Behalte Spaltennamen bei

    return tfidf_df


def determine_better_data(model1, name1, model2, name2, X1, X2, y, validate=False):
    y_score_1 = model1.predict_proba(X1)[:, 1]
    y_score_2 = model2.predict_proba(X2)[:, 1]
    precision_1, recall_1, _ = precision_recall_curve(y,y_score_1)
    precision_2, recall_2, _ = precision_recall_curve(y, y_score_2)

    auc_1 = auc(recall_1, precision_1)
    auc_2 = auc(recall_2, precision_2)

    fig, ax = plt.subplots()

    # Plot für die diagonale Linie (optional)
    ax.plot([0, 1], [0, 1], 'k--')

    # Begrenzungen setzen
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])

    # Achsenbeschriftungen
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall-Kurve zum Modellvergleich')

    # Erste PR-Kurve plotten und im selben ax zeichnen
    display_1 = PrecisionRecallDisplay(precision=precision_1, recall=recall_1)
    display_1.plot(ax=ax, label=name1 + f" (AUC = {auc_1:.2f})")

    # Zweite PR-Kurve plotten und im selben ax zeichnen
    display_2 = PrecisionRecallDisplay(precision=precision_2, recall=recall_2)
    display_2.plot(ax=ax, label=name2 + f" (AUC = {auc_2:.2f})")

    # Legende oben rechts
    ax.legend(loc='upper right')

    # Plot anzeigen
    plt.show()

    if auc_1 > auc_2:
        better_model = model1
        better_data = name1
        better_precision = precision_1
        better_recall = recall_1
        better_auc = auc_1
    else:
        better_model = model2
        better_data = name2
        better_precision = precision_2
        better_recall = recall_2
        better_auc = auc_2

    return{
        'better_data': better_data,
        'better_model': better_model,
        'better_precision': better_precision,
        'better_recall': better_recall,
        'better_auc': better_auc
    }

def find_best_model(*models):
    results = []

    for model_entry in models:
        model_name = model_entry['name']
        model = model_entry['model']
        X_test = model_entry['X_test']
        y_test = model_entry['y_test']

        # Vorhersagewahrscheinlichkeiten
        y_score = model.predict_proba(X_test)[:, 1]

        # Berechnung der Precision-Recall-Kurve
        precision, recall, _ = precision_recall_curve(y_test, y_score)

        # Fläche unter der Kurve (AUC) berechnen
        model_auc = auc(recall, precision)

        # Ergebnisse speichern
        results.append({
            'model_name': model_name,
            'precision': max(precision),
            'recall': max(recall),
            'auc': model_auc
        })

    # Sortieren nach Precision
    precision_ranking = sorted(results, key=lambda x: x['precision'], reverse=True)
    # Sortieren nach Recall
    recall_ranking = sorted(results, key=lambda x: x['recall'], reverse=True)
    # Sortieren nach AUC (Overall Ranking)
    overall_ranking = sorted(results, key=lambda x: x['auc'], reverse=True)

    return {
        'precision_ranking': precision_ranking,
        'recall_ranking': recall_ranking,
        'overall_ranking': overall_ranking
    }


def save_model(model, modelname):

    directory = 'Data\modelle'
    os.makedirs(directory, exist_ok=True)
    filename = os.path.join(directory, modelname)

    with open(filename, 'wb') as file:
        pickle.dump(model, file)

    print(f"Modell gespeichert als {filename}")



