import scipy.io
import pandas as pd
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import precision_recall_curve, auc, PrecisionRecallDisplay, classification_report, precision_score
import matplotlib.pyplot as plt
import pickle
import os
import numpy as np
from sklearn.tree import plot_tree



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

    ax.plot([0, 1], [0, 1], 'k--')

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


    if report_1["-1"]['precision'] > report_2["-1"]['precision'] and report_1["-1"]['precision'] > precision_dummy_no_spam:
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
    fig, axs = plt.subplots(2, len(models), figsize=(15, 8))  # 2 rows of plots: one for PrC and one for decision boundary

    for idx, model_entry in enumerate(models):
        model_name = model_entry['name']
        model = model_entry['model']
        X_test = model_entry['X_test']
        y_test = model_entry['y_test']

        # Handle Dummy Classifier separately
        if model_name == 'Dummy':
            # Check if predict_proba exists
            if hasattr(model, 'predict_proba'):
                y_score = model.predict_proba(X_test)[:, 1]
            else:
                # Assign probabilities based on the most frequent class
                most_frequent_class = model.predict(X_test)[0]
                y_score = np.full_like(y_test, 1.0 if most_frequent_class == 1 else 0.0)

            # Precision-Recall for dummy classifier
            precision, recall, _ = precision_recall_curve(y_test, y_score)
            model_auc = auc(recall, precision)
            y_pred = model.predict(X_test)
        else:
            # Handle models that support either predict_proba or decision_function
            if hasattr(model, 'predict_proba'):
                y_score = model.predict_proba(X_test)[:, 1]
            elif hasattr(model, 'decision_function'):
                y_score = model.decision_function(X_test)
            else:
                raise ValueError(f"Model {model_name} does not support predict_proba or decision_function")

            # Calculate Precision-Recall Curve
            precision, recall, _ = precision_recall_curve(y_test, y_score)
            model_auc = auc(recall, precision)

            # Threshold for classification (e.g., threshold = 0.5)
            threshold = 0.5
            y_pred = (y_score >= threshold).astype(int)

        # Plot the Precision-Recall curve
        axs[0, idx].plot(recall, precision, label=f'{model_name} (AUC={model_auc:.2f})')
        axs[0, idx].set_xlabel('Recall')
        axs[0, idx].set_ylabel('Precision')
        axs[0, idx].set_title(f'PrC: {model_name}')
        axs[0, idx].legend()

        # Store results
        results.append({
            'model_name': model_name,
            'precision': max(precision),
            'recall': max(recall),
            'auc': model_auc
        })

        # Plot the decision boundary for interpretable models
        if model_name != 'Dummy' and hasattr(model, 'predict'):
            plot_decision_boundary(model, X_test, y_test, axs[1, idx], model_name)

            # If the model is a DecisionTreeClassifier, plot its structure
            if hasattr(model, 'tree_'):
                fig_tree, ax_tree = plt.subplots(figsize=(12, 8))
                plot_tree(model, ax=ax_tree, filled=True, feature_names=['PCA1', 'PCA2'])  # Adjust feature names as necessary
                ax_tree.set_title(f'Decision Tree Structure: {model_name}')
                plt.show()

    # Show all visualizations
    plt.tight_layout()
    plt.show()

    # Sort models by precision, recall, and AUC
    precision_ranking = sorted(results, key=lambda x: x['precision'], reverse=True)
    recall_ranking = sorted(results, key=lambda x: x['recall'], reverse=True)
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


def plot_decision_boundary(model, X, y, ax, model_name):
    # Ensure X is in the correct format (NumPy array)
    if isinstance(X, pd.DataFrame):
        X = X.values

    # Define grid range using NumPy indexing
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))

    # Predict on grid
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plot decision boundary and test points
    ax.contourf(xx, yy, Z, alpha=0.4, cmap=plt.cm.coolwarm)
    ax.scatter(X[:, 0], X[:, 1], c=y, s=30, edgecolor='k', cmap=plt.cm.coolwarm)
    ax.set_title(f'Decision Boundary: {model_name}')