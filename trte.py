from sklearn.metrics import classification_report
from utils import save_model


def train_test(model, X_train, X_temp, y_train, y_temp, modelname):
    """
    Trainiert ein Modell mit den bereitgestellten Trainingsdaten und bewertet es anhand der Testdaten.

    :param model: Das zu trainierende Maschinenlernmodell.
    :param X_train: Die Merkmale der Trainingsdaten.
    :param X_temp: Die temporären oder Testdatenmerkmale für die Validierung.
    :param y_train: Die Labels, die den Trainingsdatenmerkmalen entsprechen.
    :param y_temp: Die Labels, die den temporären oder Testdatenmerkmalen entsprechen.
    :param modelname: Der Name, unter dem das trainierte Modell gespeichert wird.
    :return: Das trainierte Maschinenlernmodell.
    """
    model.fit(X_train, y_train)

    y_test_pred = model.predict(X_temp)

    print("Testdaten:")
    report = classification_report(y_temp, y_test_pred, digits=4, output_dict=True)
    print(" - Genauigkeit für nicht Spam: ", report['-1']['precision'], "   - Genauigkeit für Spam: ",
          report['1']['precision'])
    print(" - Abrufrate für nicht Spam:  ", report['-1']["recall"], "       - Abrufrate für Spam:  ",
          report['1']["recall"])
    print("=========================================================================================================")
    save_model(model, modelname)
    return model



def train_dummy(dummy, X, y, modelname):
    """
    Trainiert ein Dummy-Modell mit den bereitgestellten Daten und bewertet es.

    :param dummy: Das zu trainierende Dummy-Modell.
    :param X: Die Merkmale der Daten.
    :param y: Die Labels, die den Datenmerkmalen entsprechen.
    :param modelname: Der Name, unter dem das trainierte Dummy-Modell gespeichert wird.
    :return: Das trainierte Dummy-Modell.
    """
    dummy.fit(X, y)
    y_pred = dummy.predict(X)

    print("\nKlassifikationsbericht:")
    report = classification_report(y, y_pred, digits=4, output_dict=True)
    print(" - Genauigkeit für nicht Spam: ", report['-1']['precision'], "   - Genauigkeit für Spam: ",
          report['1']['precision'])
    print(" - Abrufrate für nicht Spam:  ", report['-1']["recall"], "       - Abrufrate für Spam:  ",
          report['1']["recall"])

    save_model(dummy, modelname)
    print("=========================================================================================================")
    return dummy
