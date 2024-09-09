from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def random_forest(X_train, X_test, y_train, y_test):

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    error_rate_accuracy = 1 - accuracy
    error_rate_precision = 1 - precision
    error_rate_recall = 1 - recall
    error_rate_f1 = 1 - f1
    print("SVM mit:", model.get_params())
    print(f'Accuracy: {accuracy * 100:.2f}%')
    print(f'Precision: {precision * 100:.2f}%')
    print(f'Recall: {recall * 100:.2f}%')
    print(f'F1-Score: {f1 * 100:.2f}%')
    print(f'Fehlerrate(1-Accuracy): {error_rate_accuracy * 100:.2f}%')
    print(f'Fehlerrate(1-Precision): {error_rate_precision * 100:.2f}%')
    print(f'Fehlerrate(1-Recall): {error_rate_recall * 100:.2f}%')
    print(f'Fehlerrate(1-F1-Score): {error_rate_f1 * 100:.2f}%')
