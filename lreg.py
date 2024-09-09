from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def logistic_regression(X_train, X_test, y_train, y_test):
    model = LogisticRegression(solver='liblinear', random_state=42)  # solver anpassen, falls nötig
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    error_rate = 1 - accuracy
    print("Logistische Regression mit:", model.get_params())
    print(f'Accuracy: {accuracy * 100:.2f}%')
    print(f'Fehlerrate: {error_rate * 100:.2f}%')

    # Prüfung der Fehlerrate
    if error_rate <= 0.002:  # 0.2% Fehlerquote
        print("Fehlerquote akzeptabel!")
    else:
        print("Fehlerquote zu hoch, weiteres Feintuning notwendig.")
