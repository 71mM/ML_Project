from sklearn.dummy import DummyClassifier
from utils import load_data, tf_idf, determine_better_data, find_best_model
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from hold_one_out import train_test, train_dummy
from Tripple_Crossvalidation import triple_cross_validation
import paramGrid

calculate_everything_new = True
X, y = load_data()


#
#Data Visualisation and preprocessing
#

test_size = 0.2
val_size = 0.2

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=(test_size + val_size))
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=(val_size/(test_size + val_size)))

X_train_tf_idf = tf_idf(X_train)
X_val_tf_idf = tf_idf(X_val)
X_test_tf_idf = tf_idf(X_test)

dummy = DummyClassifier(strategy="most_frequent")
decisiontree = DecisionTreeClassifier()
randomforest = RandomForestClassifier()
lineare_klassifikation = LinearSVC()
lineare_regression = LinearRegression()
logistische_regression = LogisticRegression()
model_entries = []

print("Aktuelles Training Dummy Classifier (Most Frequent):")
dummy_trte = train_dummy(dummy, X_train, X_test, y_train, y_test, "model_dummy_train_test")
dummy_tf_idf = train_dummy(dummy, X_train_tf_idf, X_test_tf_idf, y_train, y_test, "model_dummy_train_test_tf_idf")
better_results = determine_better_data(dummy_tf_idf,"mit TF-IDF Transformation",
                                       dummy_trte,"ohne TF-IDF Transformation", X_test_tf_idf, X_test, y_test)
print(f"Für das Dummy-Training, haben die Daten {better_results['better_data']} zu einem besseren Ergebnis geführt:")
print(f" - Precision: {better_results['better_precision']}")
print(f" - Recall: {better_results['better_recall']}")
print(f" - AUC: {better_results['better_auc']:.2f}")
if better_results['better_data'] == " mit TF-IDF Transformation":
    best_dummy = dummy_tf_idf
    X_test_dumm = X_test_tf_idf
else:
    best_dummy = dummy_trte
    X_test_dumm = X_test
model_entries.append({'name': 'Dummy', 'model': best_dummy, 'X_test': X_test_dumm, 'y_test': y_test})
better_results = None
print("Aktuelles Training Entscheidungsbaum")
best_decisiontree_tf_idf_trvate = train_test(decisiontree, X_train_tf_idf, X_val_tf_idf, X_test_tf_idf, y_train, y_val, y_test, "model_dt_train_test_tf_idf")
best_decisiontree_trvate = train_test(decisiontree, X_train, X_val, X_test, y_train, y_val, y_test, "model_dt_train_test")
better_results = determine_better_data(best_decisiontree_tf_idf_trvate, "mit TF-IDF Transformation" ,
                                       best_decisiontree_trvate, "ohne TF-IDF Transformation", X_test_tf_idf, X_test, y_test)

print(f"Für das Trainingsverfahren: Entscheidungsbaum, haben die Daten {better_results['better_data']} zu einem besseren Ergebnis geführt:")
print(f" - Precision: {better_results['better_precision']}")
print(f" - Recall: {better_results['better_recall']}")
print(f" - AUC: {better_results['better_auc']:.2f}")
print(f"Daher wird die Tripple Cross Validierung mit den Daten {better_results['better_data']} durchgeführt.")

if better_results['better_data'] == "mit TF-IDF Transformation":
    best_decisiontree = triple_cross_validation(decisiontree, X_train_tf_idf, X_val_tf_idf, X_test_tf_idf, y_train, y_val, y_test,
                                                paramGrid.decision_tree_grid, "model_dt_tcv_tf_idf")
    X_test_dt = X_test_tf_idf
else:
    best_decisiontree = triple_cross_validation(decisiontree, X_train, X_val, X_test, y_train, y_val, y_test,
                                                paramGrid.decision_tree_grid , "model_dt_tcv_tf_idf")
    X_test_dt = X_test
model_entries.append({'name': 'Decision Tree', 'model': best_decisiontree, 'X_test': X_test_dt, 'y_test': y_test})
better_results = None

print("Aktuelles Training Random Forest")
best_randomforest_tf_idf_trvate = train_test(randomforest, X_train_tf_idf, X_val_tf_idf, X_test_tf_idf, y_train, y_val,
                                             y_test, "model_rf_train_test_tf_idf")
best_randomforest_trvate = train_test(randomforest, X_train, X_val, X_test, y_train, y_val, y_test, "model_rf_train_test")
better_results = determine_better_data(best_randomforest_tf_idf_trvate, "mit TF-IDF Transformation",
                                          best_randomforest_trvate, "ohne TF-IDF Transformation", X_test_tf_idf, X_test,
                                          y_test)

print(f"Für das Trainingsverfahren: Random Forest, haben die Daten {better_results['better_data']} zu einem besseren Ergebnis geführt:")
print(f" - Precision: {max(better_results['better_precision'])}")
print(f" - Recall: {max(better_results['better_recall'])}")
print(f" - AUC: {better_results['better_auc']:.2f}")
print(f"Daher wird die Tripple Cross Validierung mit den Daten {better_results['better_data']} durchgeführt.")

if better_results['better_data'] == "mit TF-IDF Transformation":
    best_randomforest = triple_cross_validation(randomforest, X_train_tf_idf, X_val_tf_idf, X_test_tf_idf, y_train,
                                                    y_val, y_test, paramGrid.random_forest_grid, "model_rf_tcv_tf_idf")
    X_test_rf = X_test_tf_idf
else:
    best_randomforest = triple_cross_validation(randomforest, X_train, X_val, X_test, y_train, y_val, y_test,
                                                    paramGrid.random_forest_grid, "model_rf_tcv")
    X_test_rf = X_test
model_entries.append({'name': 'Random Forest', 'model': best_randomforest, 'X_test': X_test_rf, 'y_test': y_test})
better_results = None

print("Aktuelles Training Lineare Klassifikation")
best_lineare_klassifikation_tf_idf_trvate = train_test(lineare_klassifikation, X_train_tf_idf, X_val_tf_idf,
                                                       X_test_tf_idf, y_train, y_val, y_test, "model_lc_train_test_tf_idf")
best_lineare_klassifikation_trvate = train_test(lineare_klassifikation, X_train, X_val, X_test, y_train, y_val, y_test, "model_lc_train_test")
better_results = determine_better_data(best_lineare_klassifikation_tf_idf_trvate, "mit TF-IDF Transformation",
                                          best_lineare_klassifikation_trvate, "ohne TF-IDF Transformation",
                                          X_test_tf_idf, X_test, y_test)

print(f"Für das Trainingsverfahren: Lineare Klassifikation, haben die Daten {better_results['better_data']} zu einem besseren Ergebnis geführt:")
print(f" - Precision: {max(better_results['better_precision'])}")
print(f" - Recall: {max(better_results['better_recall'])}")
print(f" - AUC: {better_results['better_auc']:.2f}")
print(f"Daher wird die Tripple Cross Validierung mit den Daten {better_results['better_data']} durchgeführt.")

if better_results['better_data'] == "mit TF-IDF Transformation":
    best_lineare_klassifikation = triple_cross_validation(lineare_klassifikation, X_train_tf_idf, X_val_tf_idf,
                                                              X_test_tf_idf, y_train, y_val, y_test,
                                                              paramGrid.lineare_klassifikation_grid, "model_lc_tcv_tf_idf")
    X_test_lc = X_test_tf_idf
else:
    best_lineare_klassifikation = triple_cross_validation(lineare_klassifikation, X_train, X_val, X_test, y_train,
                                                              y_val, y_test, paramGrid.lineare_klassifikation_grid, "model_lc_tcv")
    X_test_lc = X_test
model_entries.append({'name': 'Lineare Klassifikation', 'model': best_lineare_klassifikation, 'X_test': X_test_lc, 'y_test': y_test})
better_results = None

print("Aktuelles Training Lineare Regression")
best_lineare_regression_tf_idf_trvate = train_test(lineare_regression, X_train_tf_idf, X_val_tf_idf, X_test_tf_idf,
                                                   y_train, y_val, y_test, "model_lr_train_test_tf_idf")
best_lineare_regression_trvate = train_test(lineare_regression, X_train, X_val, X_test, y_train, y_val, y_test, "model_lr_train_test")
better_results = determine_better_data(best_lineare_regression_tf_idf_trvate, "mit TF-IDF Transformation",
                                          best_lineare_regression_trvate, "ohne TF-IDF Transformation", X_test_tf_idf,
                                          X_test, y_test)

print(f"Für das Trainingsverfahren: Lineare Regression, haben die Daten {better_results['better_data']} zu einem besseren Ergebnis geführt:")
print(f" - Precision: {max(better_results['better_precision'])}")
print(f" - Recall: {max(better_results['better_recall'])}")
print(f" - AUC: {better_results['better_auc']:.2f}")
print(f"Daher wird die Tripple Cross Validierung mit den Daten {better_results['better_data']} durchgeführt.")

if better_results['better_data'] == "mit TF-IDF Transformation":
    best_lineare_regression = triple_cross_validation(lineare_regression, X_train_tf_idf, X_val_tf_idf,
                                                          X_test_tf_idf, y_train, y_val, y_test,
                                                          paramGrid.svm_grid, "model_lr_tcv_tf_idf")
    X_test_lr = X_test_tf_idf
else:
    best_lineare_regression = triple_cross_validation(lineare_regression, X_train, X_val, X_test, y_train, y_val,
                                                          y_test, paramGrid.svm_grid, "model_lr_tcv")
    X_test_lr = X_test
model_entries.append({'name': 'Lineare Regression', 'model': best_lineare_regression, 'X_test': X_test_lr, 'y_test': y_test})
better_results = None

print("Aktuelles Training Logistische Regression")
best_logistische_regression_tf_idf_trvate = train_test(logistische_regression, X_train_tf_idf, X_val_tf_idf,
                                                       X_test_tf_idf, y_train, y_val, y_test, "model_lor_train_test_tf_idf")
best_logistische_regression_trvate = train_test(logistische_regression, X_train, X_val, X_test, y_train, y_val, y_test, "model_lor_train_test")
better_results = determine_better_data(best_logistische_regression_tf_idf_trvate, "mit TF-IDF Transformation",
                                          best_logistische_regression_trvate, "ohne TF-IDF Transformation",
                                          X_test_tf_idf, X_test, y_test)

print(f"Für das Trainingsverfahren: Logistische Regression, haben die Daten {better_results['better_data']} zu einem besseren Ergebnis geführt:")
print(f" - Precision: {max(better_results['better_precision'])}")
print(f" - Recall: {max(better_results['better_recall'])}")
print(f" - AUC: {better_results['better_auc']:.2f}")
print(f"Daher wird die Tripple Cross Validierung mit den Daten {better_results['better_data']} durchgeführt.")

if better_results['better_data'] == "mit TF-IDF Transformation":
    best_logistische_regression = triple_cross_validation(logistische_regression, X_train_tf_idf, X_val_tf_idf,
                                                              X_test_tf_idf, y_train, y_val, y_test,
                                                              paramGrid.logistische_regression_grid, "model_lor_tcv_tf_idf")
    X_test_lreg = X_test_tf_idf
else:
    best_logistische_regression = triple_cross_validation(logistische_regression, X_train, X_val, X_test, y_train,
                                                              y_val, y_test, paramGrid.logistische_regression_grid, "model_lor_tcv")
    X_test_lreg = X_test
model_entries.append({'name': 'Logistische Regression', 'model': best_logistische_regression, 'X_test': X_test_lreg, 'y_test': y_test})
better_results = None

best_model_ranking = find_best_model(*model_entries)
print("Precision Ranking:")
for rank, entry in enumerate(best_model_ranking['precision_ranking'], start=1):
    print(f"{rank}. {entry['model_name']} - Precision: {entry['precision']:.4f}")

print("\nRecall Ranking:")
for rank, entry in enumerate(best_model_ranking['recall_ranking'], start=1):
    print(f"{rank}. {entry['model_name']} - Recall: {entry['recall']:.4f}")

print("\nOverall Ranking (AUC):")
for rank, entry in enumerate(best_model_ranking['overall_ranking'], start=1):
    print(f"{rank}. {entry['model_name']} - AUC: {entry['auc']:.4f}")

overall_best_model_entry = best_model_ranking['overall_ranking'][0]
overall_best_model = overall_best_model_entry['model']
overall_best_params = overall_best_model.best_params_
print(f"\nOverall Best Model: {overall_best_model_entry['model_name']}")
print("Beste Parameter:")
for param in overall_best_params.keys():
    print(f'{param}: {overall_best_params[param]}')


