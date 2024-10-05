from sklearn.dummy import DummyClassifier
from utils import load_data, tf_idf, determine_better_data, find_best_model
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from hold_one_out import train_test, train_dummy
from Tripple_Crossvalidation import triple_cross_validation
import paramGrid
import matplotlib.pyplot as plt
import os
import seaborn as sns


calculate_everything_new = True
X, y = load_data()

print(X.head())
print(y.head())
print(X.tail())
print(y.tail())

# Check for missing values
print(X.isnull().sum())
print(y.isnull().sum())

# Display the shape of the datasets
print(X.shape)
print(y.shape)

visualize = X
visualize['kategorie'] = y
visualize = visualize.groupby('kategorie').sum()
# Erstelle die Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(visualize, annot=True, fmt="d", cmap="YlGnBu", linewidths=.5)


plt.title('Wort-Häufigkeit Heatmap')
plt.xlabel('Wörter')
plt.ylabel('E-Mails')
plt.show()
save_folder = 'Data/bilder'
plt.savefig(os.path.join(save_folder, 'Daten_visualization.pdf'))
input()

test_size = 0.2
val_size = 0.2

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=(test_size + val_size))
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=(val_size/(test_size + val_size)))

X_train_tf_idf = tf_idf(X_train)
X_temp_tf_idf = tf_idf(X_temp)
X_val_tf_idf = tf_idf(X_val)
X_test_tf_idf = tf_idf(X_test)

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

print("Aktuelles Training Dummy Classifier (Most Frequent):")
dummy_trte = train_dummy(dummy, X, y, "model_dummy_train_test")
better_results = determine_better_data(None, None, None, None, None, X_temp, y_temp, dummy=dummy_trte)
print(f"Für das Dummy-Training, haben die Daten {better_results['better_data']} zu einem besseren Ergebnis geführt:")
print(f" - Precision: {better_results['better_precision'][0]}")
print(f" - Recall: {better_results['better_recall'][0]}")
print(f" - AUC: {better_results['better_auc']:.2f}")
if better_results['better_data'] == " mit TF-IDF Transformation":
    best_dummy = dummy_trte
    X_test_dumm = X_test
else:
    best_dummy = dummy_trte
    X_test_dumm = X_test
model_entries.append({'name': 'Dummy', 'model': best_dummy, 'X_test': X_test_dumm, 'y_test': y_test})
better_results = None


print("Aktuelles Training Entscheidungsbaum")
best_decisiontree_tf_idf_trvate = train_test(decisiontree_1, X_train_tf_idf, X_test_tf_idf, y_train, y_test, "model_dt_train_test_tf_idf")
best_decisiontree_trvate = train_test(decisiontree_2, X_train, X_test, y_train, y_test, "model_dt_train_test")
better_results = determine_better_data(best_decisiontree_tf_idf_trvate, " Decision Tree mit TF-IDF Transformation", best_decisiontree_trvate, "Decision Tree ohne TF-IDF Transformation",
                                       X_test_tf_idf, X_test, y_test, dummy_trte)


print(f"Für das Trainingsverfahren: Entscheidungsbaum, haben die Daten {better_results['better_data']} zu einem besseren Ergebnis geführt:")
print(f" - Precision für nicht Spam : {better_results['better_precision'][0]}      - Precision für Spam: {better_results['better_precision'][1]}")
print(f" - Recall für nicht Spam : {better_results['better_recall'][0]}           - Recall für Spam: {better_results['better_recall'][1]}")
print(f" - AUC: {better_results['better_auc']:.2f}")
print(f"Daher wird die Tripple Cross Validierung mit den Daten {better_results['better_data']} durchgeführt.")

if better_results['better_data'] == "Decision Tree mit TF-IDF Transformation":
    best_decisiontree = triple_cross_validation(decisiontree_1, X_train_tf_idf, X_val_tf_idf, X_test_tf_idf, y_train, y_val, y_test,
                                                paramGrid.decision_tree_grid, "model_dt_tcv_tf_idf")
    X_test_dt = X_test_tf_idf
else:
    best_decisiontree = triple_cross_validation(decisiontree_2, X_train, X_val, X_test, y_train, y_val, y_test,
                                                paramGrid.decision_tree_grid, "model_dt_tcv_tf_idf")
    X_test_dt = X_test
model_entries.append({'name': 'Decision Tree', 'model': best_decisiontree, 'X_test': X_test_dt, 'y_test': y_test})
better_results = None

print("Aktuelles Training Random Forest")
best_randomforest_tf_idf_trvate = train_test(randomforest_1, X_train_tf_idf, X_test_tf_idf, y_train, y_test, "model_rf_train_test_tf_idf")
best_randomforest_trvate = train_test(randomforest_2, X_train, X_test, y_train, y_test, "model_rf_train_test")
better_results = determine_better_data(best_randomforest_tf_idf_trvate, "Random Forest mit TF-IDF Transformation",
                                          best_randomforest_trvate, "Random Forest ohne TF-IDF Transformation", X_test_tf_idf, X_test,
                                          y_test, dummy_trte)


print(f"Für das Trainingsverfahren: Random Forest, haben die Daten {better_results['better_data']} zu einem besseren Ergebnis geführt:")
print(f" - Precision für nicht Spam : {better_results['better_precision'][0]}     - Precision für Spam: {better_results['better_precision'][1]}")
print(f" - Recall für nicht Spam : {better_results['better_recall'][0]}           - Recall für Spam: {better_results['better_recall'][1]}")
print(f" - AUC: {better_results['better_auc']:.2f}")
print(f"Daher wird die Tripple Cross Validierung mit den Daten {better_results['better_data']} durchgeführt.")

if better_results['better_data'] == "Random Forest mit TF-IDF Transformation":
    best_randomforest = triple_cross_validation(randomforest_1, X_train_tf_idf, X_val_tf_idf, X_test_tf_idf, y_train, y_val, y_test, paramGrid.random_forest_grid, "model_rf_tcv_tf_idf")
    X_test_rf = X_test_tf_idf
else:
    best_randomforest = triple_cross_validation(randomforest_2, X_train, X_val, X_test, y_train, y_val, y_test, paramGrid.random_forest_grid, "model_rf_tcv")
    X_test_rf = X_test
model_entries.append({'name': 'Random Forest', 'model': best_randomforest, 'X_test': X_test_rf, 'y_test': y_test})
better_results = None

print("Aktuelles Training Lineare Klassifikation")
best_lineare_klassifikation_tf_idf_trvate = train_test(lineare_klassifikation_1, X_train_tf_idf, X_test_tf_idf, y_train, y_test, "model_lc_train_test_tf_idf")
best_lineare_klassifikation_trvate = train_test(lineare_klassifikation_2, X_train, X_test, y_train, y_test, "model_lc_train_test")
better_results = determine_better_data(best_lineare_klassifikation_tf_idf_trvate, "Lineare Klassifikation mit TF-IDF Transformation",
                                          best_lineare_klassifikation_trvate, "Lineare Klassifikation ohne TF-IDF Transformation",
                                          X_test_tf_idf, X_test, y_test, dummy_trte)


print(f"Für das Trainingsverfahren: Lineare Klassifikation, haben die Daten {better_results['better_data']} zu einem besseren Ergebnis geführt:")
print(f" - Precision für nicht Spam : {better_results['better_precision'][0]}     - Precision für Spam: {better_results['better_precision'][1]}")
print(f" - Recall für nicht Spam : {better_results['better_recall'][0]}           - Recall für Spam: {better_results['better_recall'][1]}")
print(f" - AUC: {better_results['better_auc']:.2f}")
print(f"Daher wird die Tripple Cross Validierung mit den Daten {better_results['better_data']} durchgeführt.")

if better_results['better_data'] == "Lineare Klassifikation mit TF-IDF Transformation":
    best_lineare_klassifikation = triple_cross_validation(lineare_klassifikation_1, X_train_tf_idf, X_val_tf_idf,
                                                              X_test_tf_idf, y_train, y_val, y_test,
                                                              paramGrid.lineare_klassifikation_grid, "model_lc_tcv_tf_idf")
    X_test_lc = X_test_tf_idf
else:
    best_lineare_klassifikation = triple_cross_validation(lineare_klassifikation_2, X_train, X_val, X_test, y_train,
                                                              y_val, y_test, paramGrid.lineare_klassifikation_grid, "model_lc_tcv")
    X_test_lc = X_test
model_entries.append({'name': 'Lineare Klassifikation', 'model': best_lineare_klassifikation, 'X_test': X_test_lc, 'y_test': y_test})
better_results = None




print("Aktuelles Training Logistische Regression")
best_logistische_regression_tf_idf_trvate = train_test(logistische_regression_1, X_train_tf_idf, X_test_tf_idf, y_train, y_test, "model_lor_train_test_tf_idf")
best_logistische_regression_trvate = train_test(logistische_regression_2, X_train, X_test, y_train, y_test, "model_lor_train_test")
better_results = determine_better_data(best_logistische_regression_tf_idf_trvate, "Logistische Regression mit TF-IDF Transformation",
                                          best_logistische_regression_trvate, "Logistische Regression ohne TF-IDF Transformation",
                                          X_test_tf_idf, X_test, y_test, dummy_trte)

print(f"Für das Trainingsverfahren: Logistische Regression, haben die Daten {better_results['better_data']} zu einem besseren Ergebnis geführt:")
print(f" - Precision für nicht Spam : {better_results['better_precision'][0]}     - Precision für Spam: {better_results['better_precision'][1]}")
print(f" - Recall für nicht Spam : {better_results['better_recall'][0]}           - Recall für Spam: {better_results['better_recall'][1]}")
print(f" - AUC: {better_results['better_auc']:.2f}")
print(f"Daher wird die Tripple Cross Validierung mit den Daten {better_results['better_data']} durchgeführt.")

if better_results['better_data'] == "Logistische Regression mit TF-IDF Transformation":
    best_logistische_regression = triple_cross_validation(logistische_regression_1, X_train_tf_idf, X_val_tf_idf,
                                                              X_test_tf_idf, y_train, y_val, y_test,
                                                              paramGrid.logistische_regression_grid, "model_lor_tcv_tf_idf")
    X_test_lreg = X_test_tf_idf
else:
    best_logistische_regression = triple_cross_validation(logistische_regression_2, X_train, X_val, X_test, y_train,
                                                              y_val, y_test, paramGrid.logistische_regression_grid, "model_lor_tcv")
    X_test_lreg = X_test
model_entries.append({'name': 'Logistische Regression', 'model': best_logistische_regression, 'X_test': X_test_lreg, 'y_test': y_test})
better_results = None

best_model_ranking = find_best_model(*model_entries)
print("Precision Ranking:")
for rank, entry in enumerate(best_model_ranking['precision_ranking'], start=1):
    print(f"{rank}. {entry['model_name']} - Precision nicht Spam: {entry['precision'][0]:.4f} - Precision Spam: {entry['precision'][1]:.4f}")
print("\nRecall Ranking:")
for rank, entry in enumerate(best_model_ranking['recall_ranking'], start=1):
    print(f"{rank}. {entry['model_name']} - Recall nicht Spam: {entry['recall'][0]:.4f} - Recall Spam: {entry['recall'][1]:.4f}")

print("\nOverall Ranking (AUC):")
for rank, entry in enumerate(best_model_ranking['overall_ranking'], start=1):
    print(f"{rank}. {entry['model_name']} - AUC: {entry['auc']:.4f}")

overall_best_model_entry = best_model_ranking['overall_ranking'][0]
overall_best_model = overall_best_model_entry['model']
overall_best_params = overall_best_model.get_params()
print(f"\nOverall Best Model: {overall_best_model_entry['model_name']}")
print("Beste Parameter:")
for param in overall_best_params.keys():
    print(f'{param}: {overall_best_params[param]}')

y_test_pred = overall_best_model_entry.predict(X_test)
report = classification_report(y_test, y_test_pred, digits=4, output_dict=True)
print(" - Precision für nicht Spam : ", report['-1']['precision'], "   - Precision für Spam: ", report['1']['precision'])
print(" - Recall für nicht Spam :  ", report['-1']["recall"], "       - Recall für Spam:  ", report['1']["recall"])

