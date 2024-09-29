from sklearn.metrics import classification_report
from utils import save_model


def train_test(model, X_train, X_val, X_test, y_train, y_val, y_test, modelname):

    model.fit(X_train, y_train)

    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)

    print("Validierungsdaten:")
    print(classification_report(y_val, y_val_pred))
    print("Testdaten:")
    print(classification_report(y_test, y_test_pred))
    print("=========================================================================================================")
    save_model(model, modelname)
    return model


def train_dummy(dummy, X_train, X_test, y_train, y_test, modelname):
    dummy.fit(X_train, y_train)
    y_pred = dummy.predict(X_test)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    save_model(dummy, modelname)
    print("=========================================================================================================")
    return dummy