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


def determine_better_data(model1, name1, model2, name2, X1, X2, y, dummy, dummy_name="Baseline Model"):

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

    # Begrenzungen setzen
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])

    # Achsenbeschriftungen
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

    # Legende oben rechts
    ax.legend(loc='best', bbox_to_anchor=(1, 1))

    # Plot anzeigen
    save_folder = 'Data/bilder'
    plt.savefig(os.path.join(save_folder, f'{name1}_visualization.pdf'))
    plt.show()
    if name1 is not None:
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
        print("Das Beste Model ist das Baseline-Modell.")
        return {
            'better_data': better_data,
            'better_model': better_model,
            'better_precision': better_precision,
            'better_recall': better_recall,
            'better_auc': better_auc
        }

    precision_dummy_no_spam = precision_score(y, y_score_dummy, zero_division=0)


    if report_1["-1"]['precision'] >= report_2["-1"]['precision'] and report_1["-1"]['precision'] > precision_dummy_no_spam:
        better_model = model1
        better_data = name1
        better_precision = [report_1["-1"]['precision'], report_1["1"]['precision']]
        better_recall = [report_1["-1"]['recall'], report_1["1"]['recall']]
        better_auc = auc_1

    elif report_2["-1"]['precision'] > report_1["-1"]['precision'] and report_2["-1"]['precision'] > precision_dummy_no_spam:
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
        print("Das Beste Model ist das Baseline-Modell.")

    return{
        'better_data': better_data,
        'better_model': better_model,
        'better_precision': better_precision,
        'better_recall': better_recall,
        'better_auc': better_auc
    }


def find_best_model(*models):
    results = []
    fig, axs = plt.subplots(2, len(models), figsize=(15, 12))

    for idx, model_entry in enumerate(models):
        model_name = model_entry['name']
        model = model_entry['model']
        X_test = model_entry['X_test']
        y_test = model_entry['y_test']

        y_pred = model.predict(X_test)

        y_score = np.where(y_pred == -1, -1, 1)


        precision, recall, _ = precision_recall_curve(y_test, y_score)
        model_auc = auc(recall, precision)


        axs[0, idx].plot(recall, precision, label=f'{model_name} (AUC={model_auc:.2f})')
        axs[0, idx].set_xlabel('Recall')
        axs[0, idx].set_ylabel('Precision')
        axs[0, idx].set_title(f'PrC: {model_name}')
        axs[0, idx].legend()

        # Plot the Confusion Matrix
        cm = confusion_matrix(y_test, y_pred, labels=[-1, 1])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axs[1, idx], xticklabels=['nicht Spam', 'Spam'],
                    yticklabels=['nicht Spam', 'Spam'])
        axs[1, idx].set_xlabel('Predicted')
        axs[1, idx].set_ylabel('Actual')
        axs[1, idx].set_title(f'Confusion Matrix: {model_name}')

        report = classification_report(y_test, y_pred, digits=4, output_dict=True)
        results.append({
            'model_name': model_name,
            'precision': [report['-1']['precision'], report['1']['precision']],
            'recall': [report['-1']["recall"], report['1']["recall"]],
            'auc': model_auc,
            'model': model
        })


    plt.tight_layout()
    save_folder = 'Data/bilder'
    plt.savefig(os.path.join(save_folder, f'{model_name}_visualization.pdf'))
    plt.show()

    precision_ranking = sorted(results, key=lambda x: x['precision'][0], reverse=True)
    recall_ranking = sorted(results, key=lambda x: x['recall'][0], reverse=True)
    overall_ranking = sorted(results, key=lambda x: x['auc'], reverse=True)

    return {
        'precision_ranking': precision_ranking,
        'recall_ranking': recall_ranking,
        'overall_ranking': overall_ranking
    }

def save_model(model, modelname):
    directory = 'Data/modelle'  # Ensure the directory path is correct
    os.makedirs(directory, exist_ok=True)
    filename = os.path.join(directory, modelname)

    with open(filename, 'wb') as file:
        pickle.dump(model, file)

    print(f"Model saved as {filename}")
