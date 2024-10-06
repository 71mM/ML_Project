from sklearn.dummy import DummyClassifier
from utils import load_data, tf_idf, determine_better_data, find_best_model
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from trte import train_test, train_dummy
from Tripple_Crossvalidation import triple_cross_validation
import paramGrid
import matplotlib.pyplot as plt
import os
import numpy as np
from wordcloud import WordCloud
import pandas as pd
import seaborn as sns

# Laden der Daten
X, y = load_data()

# Anzeigen der ersten und letzten Zeilen der Daten
print(X.head())
print(y.head())
print(X.tail())
print(y.tail())

# Überprüfen auf fehlende Werte
print(X.isnull().sum())
print(y.isnull().sum())

# Ausgeben der Form der Daten
print(X.shape)
print(y.shape)

# Indizes für Spam/Nicht-Spam
spam_indices = np.where(y == 1)[0]
not_spam_indices = np.where(y == -1)[0]

# Zählen der Wörter für Spam/Nicht-Spam
spam_word_counts = np.array(X.iloc[spam_indices].sum(axis=0)).flatten()
not_spam_word_counts = np.array(X.iloc[not_spam_indices].sum(axis=0)).flatten()

num_features = X.shape[1]

# Platzhalter für Wörter (falls keine echten Wörter verfügbar sind)
placeholder_words = [f'word{i}' for i in range(num_features)]

# Erzeugen der Wortfrequenz-Distributionen
spam_word_freq = {placeholder_words[idx]: spam_word_counts[idx] for idx in range(num_features)}
not_spam_word_freq = {placeholder_words[idx]: not_spam_word_counts[idx] for idx in range(num_features)}

# Erstellen der Wortwolken
spam_wc = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(spam_word_freq)
not_spam_wc = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(not_spam_word_freq)

# Plot für Wortwolken
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(spam_wc, interpolation='bilinear')
plt.title('Spam Wordcloud')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(not_spam_wc, interpolation='bilinear')
plt.title('Not Spam Wordcloud')
plt.axis('off')

plt.tight_layout()
save_folder = 'Data/bilder'
if not os.path.exists(save_folder):
    os.makedirs(save_folder)
plt.savefig(os.path.join(save_folder, 'wordcloud_visualization.pdf'))
plt.show()

# Länge der Emails
email_lengths = X.sum(axis=1).values

# TF-IDF Transformation
X_tfidf = tf_idf(X)
tfidf_sums = X_tfidf.sum(axis=1).values

# Scatterplots erstellen
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 12))

sns.scatterplot(ax=axes[0], x=np.arange(len(email_lengths)), y=email_lengths, hue=y, palette={1: 'red', -1: 'blue'},
                legend='full')
axes[0].set_title('Scatterplot der E-Mail-Längen, gefärbt nach Spam/Nicht-Spam')
axes[0].set_xlabel('E-Mail Index')
axes[0].set_ylabel('E-Mail Länge (Wörteranzahl)')
axes[0].legend(title='Kategorie', loc='upper right', labels=['Nicht-Spam', 'Spam'])

sns.scatterplot(ax=axes[1], x=np.arange(len(tfidf_sums)), y=tfidf_sums, hue=y, palette={1: 'red', -1: 'blue'})
axes[1].set_title('Scatterplot der TF-IDF-Summen, gefärbt nach Spam/Nicht-Spam')
axes[1].set_xlabel('E-Mail Index')
axes[1].set_ylabel('Summe der TF-IDF Werte')
axes[1].legend(title='Kategorie', loc='upper right', labels=['Nicht-Spam', 'Spam'])

plt.tight_layout()
save_folder = 'Data/bilder'
if not os.path.exists(save_folder):
    os.makedirs(save_folder)
plt.savefig(os.path.join(save_folder, 'scatter_visualization.pdf'))
plt.show()

# Top 10 Wörter für Spam/Nicht-Spam
spam_top_10 = sorted(spam_word_freq.items(), key=lambda x: x[1], reverse=True)[:10]
not_spam_top_10 = sorted(not_spam_word_freq.items(), key=lambda x: x[1], reverse=True)[:10]

spam_ranking_df = pd.DataFrame(spam_top_10, columns=['Word', 'Frequency'])
not_spam_ranking_df = pd.DataFrame(not_spam_top_10, columns=['Word', 'Frequency'])

print("Top 10 Spam-Wörter nach Länge:")
print(spam_ranking_df)
print("Top 10 Nicht-Spam-Wörter nach Länge:")
print(not_spam_ranking_df)

# Aufteilen der Daten in Trainings-/Validierungs-/Test-Sets
test_size = 0.2
val_size = 0.2
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=(test_size + val_size))
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=(val_size / (test_size + val_size)))

# TF-IDF Transformation der geteilten Daten
X_train_tf_idf = tf_idf(X_train)
X_temp_tf_idf = tf_idf(X_temp)
X_val_tf_idf = tf_idf(X_val)
X_test_tf_idf = tf_idf(X_test)

# Initialisieren der Modelle
dummy = DummyClassifier(strategy="most_frequent")
decisiontree_1 = DecisionTreeClassifier()
decisiontree_2 = DecisionTreeClassifier()
randomforest_1 = RandomForestClassifier()
randomforest_2 = RandomForestClassifier()
lineare_klassifikation_1 = LinearSVC()
lineare_klassifikation_2 = LinearSVC()
logistische_regression_1 = LogisticRegression()
logistische_regression_2 = LogisticRegression()

model_entries = []

# Training und Ergebnisanzeige: Dummy Classifier
print("Aktuelles Training: Dummy Classifier (Most Frequent):")
dummy_trte = train_dummy(dummy, X, y, "model_dummy_train_test")
better_results = determine_better_data(None, None, None, None, None, X_temp, y_temp, dummy=dummy_trte)
if better_results['better_data'] == " mit TF-IDF Transformation":
    best_dummy = dummy_trte
    X_test_dumm = X_test
else:
    best_dummy = dummy_trte
    X_test_dumm = X_test
model_entries.append({'name': 'Dummy', 'model': best_dummy, 'X_test': X_test_dumm, 'y_test': y_test})
better_results = None

# Training und Ergebnisanzeige: Entscheidungsbaum
print("Aktuelles Training: Entscheidungsbaum")
best_decisiontree_tf_idf_trvate = train_test(decisiontree_1, X_train_tf_idf, X_test_tf_idf, y_train, y_test,
                                             "model_dt_train_test_tf_idf")
best_decisiontree_trvate = train_test(decisiontree_2, X_train, X_test, y_train, y_test, "model_dt_train_test")
better_results = determine_better_data(best_decisiontree_tf_idf_trvate, " Decision Tree mit TF-IDF Transformation",
                                       best_decisiontree_trvate, "Decision Tree ohne TF-IDF Transformation",
                                       X_test_tf_idf, X_test, y_test, dummy_trte)
if better_results['better_data'] == "Decision Tree mit TF-IDF Transformation":
    best_decisiontree = triple_cross_validation(decisiontree_1, X_train_tf_idf, X_val_tf_idf, X_test_tf_idf, y_train,
                                                y_val, y_test,
                                                paramGrid.decision_tree_grid, "model_dt_tcv_tf_idf")
    X_test_dt = X_test_tf_idf
else:
    best_decisiontree = triple_cross_validation(decisiontree_2, X_train, X_val, X_test, y_train, y_val, y_test,
                                                paramGrid.decision_tree_grid, "model_dt_tcv_tf_idf")
    X_test_dt = X_test
model_entries.append({'name': 'Decision Tree', 'model': best_decisiontree, 'X_test': X_test_dt, 'y_test': y_test})
better_results = None

# Training und Ergebnisanzeige: Random Forest
print("Aktuelles Training: Random Forest")
best_randomforest_tf_idf_trvate = train_test(randomforest_1, X_train_tf_idf, X_test_tf_idf, y_train, y_test,
                                             "model_rf_train_test_tf_idf")
best_randomforest_trvate = train_test(randomforest_2, X_train, X_test, y_train, y_test, "model_rf_train_test")
better_results = determine_better_data(best_randomforest_tf_idf_trvate, "Random Forest mit TF-IDF Transformation",
                                       best_randomforest_trvate, "Random Forest ohne TF-IDF Transformation",
                                       X_test_tf_idf, X_test,
                                       y_test, dummy_trte)
if better_results['better_data'] == "Random Forest mit TF-IDF Transformation":
    best_randomforest = triple_cross_validation(randomforest_1, X_train_tf_idf, X_val_tf_idf, X_test_tf_idf, y_train,
                                                y_val, y_test, paramGrid.random_forest_grid, "model_rf_tcv_tf_idf")
    X_test_rf = X_test_tf_idf
else:
    best_randomforest = triple_cross_validation(randomforest_2, X_train, X_val, X_test, y_train, y_val, y_test,
                                                paramGrid.random_forest_grid, "model_rf_tcv")
    X_test_rf = X_test
model_entries.append({'name': 'Random Forest', 'model': best_randomforest, 'X_test': X_test_rf, 'y_test': y_test})
better_results = None

# Training und Ergebnisanzeige: Lineare Klassifikation
print("Aktuelles Training: Lineare Klassifikation")
best_lineare_klassifikation_tf_idf_trvate = train_test(lineare_klassifikation_1, X_train_tf_idf, X_test_tf_idf, y_train,
                                                       y_test, "model_lc_train_test_tf_idf")
best_lineare_klassifikation_trvate = train_test(lineare_klassifikation_2, X_train, X_test, y_train, y_test,
                                                "model_lc_train_test")
better_results = determine_better_data(best_lineare_klassifikation_tf_idf_trvate,
                                       "Lineare Klassifikation mit TF-IDF Transformation",
                                       best_lineare_klassifikation_trvate,
                                       "Lineare Klassifikation ohne TF-IDF Transformation",
                                       X_test_tf_idf, X_test, y_test, dummy_trte)
if better_results['better_data'] == "Lineare Klassifikation mit TF-IDF Transformation":
    best_lineare_klassifikation = triple_cross_validation(lineare_klassifikation_1, X_train_tf_idf, X_val_tf_idf,
                                                          X_test_tf_idf, y_train, y_val, y_test,
                                                          paramGrid.lineare_klassifikation_grid, "model_lc_tcv_tf_idf")
    X_test_lc = X_test_tf_idf
else:
    best_lineare_klassifikation = triple_cross_validation(lineare_klassifikation_2, X_train, X_val, X_test, y_train,
                                                          y_val, y_test, paramGrid.lineare_klassifikation_grid,
                                                          "model_lc_tcv")
    X_test_lc = X_test
model_entries.append(
    {'name': 'Lineare Klassifikation', 'model': best_lineare_klassifikation, 'X_test': X_test_lc, 'y_test': y_test})
better_results = None

# Training und Ergebnisanzeige: Logistische Regression
print("Aktuelles Training: Logistische Regression")
best_logistische_regression_tf_idf_trvate = train_test(logistische_regression_1, X_train_tf_idf, X_test_tf_idf, y_train,
                                                       y_test, "model_lor_train_test_tf_idf")
best_logistische_regression_trvate = train_test(logistische_regression_2, X_train, X_test, y_train, y_test,
                                                "model_lor_train_test")
better_results = determine_better_data(best_logistische_regression_tf_idf_trvate,
                                       "Logistische Regression mit TF-IDF Transformation",
                                       best_logistische_regression_trvate,
                                       "Logistische Regression ohne TF-IDF Transformation",
                                       X_test_tf_idf, X_test, y_test, dummy_trte)
if better_results['better_data'] == "Logistische Regression mit TF-IDF Transformation":
    best_logistische_regression = triple_cross_validation(logistische_regression_1, X_train_tf_idf, X_val_tf_idf,
                                                          X_test_tf_idf, y_train, y_val, y_test,
                                                          paramGrid.logistische_regression_grid, "model_lor_tcv_tf_idf")
    X_test_lreg = X_test_tf_idf
else:
    best_logistische_regression = triple_cross_validation(logistische_regression_2, X_train, X_val, X_test, y_train,
                                                          y_val, y_test, paramGrid.logistische_regression_grid,
                                                          "model_lor_tcv")
    X_test_lreg = X_test
model_entries.append(
    {'name': 'Logistische Regression', 'model': best_logistische_regression, 'X_test': X_test_lreg, 'y_test': y_test})
better_results = None

# Finden des besten Modells
best_model_ranking = find_best_model(*model_entries)
print("Precision Ranking:")
for rank, entry in enumerate(best_model_ranking['precision_ranking'], start=1):
    print(
        f"{rank}. {entry['model_name']} - Precision nicht Spam: {entry['precision'][0]:.4f} - Precision Spam: {entry['precision'][1]:.4f}")
print("\nRecall Ranking:")
for rank, entry in enumerate(best_model_ranking['recall_ranking'], start=1):
    print(
        f"{rank}. {entry['model_name']} - Recall nicht Spam: {entry['recall'][0]:.4f} - Recall Spam: {entry['recall'][1]:.4f}")

print("\nAUC :")
for rank, entry in enumerate(best_model_ranking['overall_ranking'], start=1):
    print(f"{rank}. {entry['model_name']} - AUC: {entry['auc']:.4f}")

# Ausgabe des besten Modells insgesamt
overall_best_model_entry = best_model_ranking['precision_ranking'][0]
overall_best_model = overall_best_model_entry['model']
overall_best_params = overall_best_model.get_params()
print(f"\nOverall Best Model: {overall_best_model_entry['model_name']}")
print("Beste Parameter:")
for param in overall_best_params.keys():
    print(f'{param}: {overall_best_params[param]}')

# Auswertung des besten Modells
y_test_pred = overall_best_model.predict(X_test)
report = classification_report(y_test, y_test_pred, digits=4, output_dict=True)
print(" - Precision für nicht Spam : ", report['-1']['precision'], "   - Precision für Spam: ",
      report['1']['precision'])
print(" - Recall für nicht Spam :  ", report['-1']["recall"], "       - Recall für Spam:  ", report['1']["recall"])