from sklearn.metrics import classification_report
from utils import save_model


def train_test(model, X_train, X_temp, y_train, y_temp, modelname):

    model.fit(X_train, y_train)

    y_test_pred = model.predict(X_temp)

    print("Testdaten:")
    report = classification_report(y_temp, y_test_pred, digits=4, output_dict=True)
    print(" - Precision für nicht Spam : ", report['-1']['precision'], "   - Precision für Spam: ", report['1']['precision'])
    print(" - Recall für nicht Spam :  ", report['-1']["recall"], "       - Recall für Spam:  ", report['1']["recall"])
    print("=========================================================================================================")
    save_model(model, modelname)
    return model


def train_dummy(dummy, X, y, modelname):
    dummy.fit(X, y)
    y_pred = dummy.predict(X)

    print("\nClassification Report:")
    report = classification_report(y, y_pred, digits=4, output_dict=True)
    print(" - Precision für nicht Spam : ", report['-1']['precision'], "   - Precision für Spam: ", report['1']['precision'])
    print(" - Recall für nicht Spam :  ", report['-1']["recall"], "       - Recall für Spam:  ", report['1']["recall"])

    save_model(dummy, modelname)
    print("=========================================================================================================")
    return dummy
