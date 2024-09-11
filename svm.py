from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import KFold, cross_validate


def svm(X_train, X_test, X_val, y_train, y_test, y_val, random_state, n_splits):

    model = SVC(kernel='linear', random_state=42)
    kfold = KFold(n_splits=n_splits, random_state=random_state, shuffle=True)
    scoring = ['accuracy', 'precision', 'recall', 'f1']
    cv_results = cross_validate(model, X_train, y_train, cv=kfold, scoring=scoring)

    print("Cross-Validation Ergebnisse (Train):")
    print(f"Accuracy: {cv_results['test_accuracy'].mean():.4f}")
    print(f"Precision: {cv_results['test_precision'].mean():.4f}")
    print(f"Recall: {cv_results['test_recall'].mean():.4f}")
    print(f"F1-Score: {cv_results['test_f1'].mean():.4f}")

    model.fit(X_train, y_train)

    y_val_pred = model.predict(X_val)

    val_accuracy = accuracy_score(y_val, y_val_pred)
    val_precision = precision_score(y_val, y_val_pred)
    val_recall = recall_score(y_val, y_val_pred)
    val_f1 = f1_score(y_val, y_val_pred)

    # Ausgabe der Metriken für Validierungsdaten
    print("\n==== Validierungsergebnisse ====")
    print(f'Accuracy (Validation): {val_accuracy * 100:.2f}%')
    print(f'Precision (Validation): {val_precision * 100:.2f}%')
    print(f'Recall (Validation): {val_recall * 100:.2f}%')
    print(f'F1-Score (Validation): {val_f1 * 100:.2f}%')

    # Vorhersagen auf den Testdaten
    y_test_pred = model.predict(X_test)

    # Berechnung der Metriken auf den Testdaten
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_precision = precision_score(y_test, y_test_pred)
    test_recall = recall_score(y_test, y_test_pred)
    test_f1 = f1_score(y_test, y_test_pred)

    # Ausgabe der Metriken für Testdaten
    print("\n==== Testergebnisse ====")
    print(f'Accuracy (Test): {test_accuracy * 100:.2f}%')
    print(f'Precision (Test): {test_precision * 100:.2f}%')
    print(f'Recall (Test): {test_recall * 100:.2f}%')
    print(f'F1-Score (Test): {test_f1 * 100:.2f}%')


