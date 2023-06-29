import pandas as pd
import seaborn as sns
import warnings

from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

cancer = pd.read_csv('C:/Users/Utilizador/Documents/Faculdade/2Semestre/IA/Cancer_Data.csv')
cancer.sample(10)

cancer.info()

cancer.drop(['id', 'Unnamed: 32'], axis=1, inplace=True)

cancer.info()

cancer_col = cancer.columns.values

cancer.head()

X = cancer.drop('diagnosis', axis='columns')
y = cancer.diagnosis

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=101)
X_valid, X_test, y_valid, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=42)

print(f'Total # of sample in whole dataset: {len(X)}')
print(f'Total # of sample in train dataset: {len(X_train)}')
print(f'Total # of sample in validation dataset: {len(X_valid)}')
print(f'Total # of sample in test dataset: {len(X_test)}')

sc = StandardScaler()

X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

model = {'DecisionTreeClassifier': DecisionTreeClassifier()}

modelName = {'DecisionTreeClassifier'}

trainScores = []
validationScores = []
testScores = []

for m in model:
    model = model[m]
    model.fit(X_train, y_train)
    score = model.score(X_valid, y_valid)

    print(f'{m}')
    train_score = model.score(X_train, y_train)
    print(f'Train score of trained model: {train_score * 100}')
    trainScores.append(train_score * 100)

    validation_score = model.score(X_valid, y_valid)
    print(f'Validation score of trained model: {validation_score * 100}')
    validationScores.append(validation_score * 100)

    test_score = model.score(X_test, y_test)
    print(f'Test score of trained model: {test_score * 100}')
    testScores.append(test_score * 100)
    print(" ")

    y_predictions = model.predict(X_test)
    conf_matrix = confusion_matrix(y_predictions, y_test)

    print(f'Confussion Matrix: \n{conf_matrix}\n')

    predictions = model.predict(X_test)
    cm = confusion_matrix(predictions, y_test)

    tn = conf_matrix[0, 0]
    fp = conf_matrix[0, 1]
    tp = conf_matrix[1, 1]
    fn = conf_matrix[1, 0]
    accuracy = (tp + tn) / (tp + fp + tn + fn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1score = 2 * precision * recall / (precision + recall)
    specificity = tn / (tn + fp)
    print(f'Accuracy : {accuracy}')
    print(f'Precision: {precision}')
    print(f'Recall   : {recall}')
    print(f'F1 score : {f1score}')
    print(f'Specificity : {specificity}') # calcula a percentagem de negativos bem identificados de todos os negativos existentes
    print("")
    print(f'Classification Report: \n{classification_report(predictions, y_test)}\n')
    print("")

    modelName = list(modelName)
    modelName.remove('DecisionTreeClassifier')

    preds = model.predict(X_test)
    confusion_matr = confusion_matrix(y_test, preds)  # normalize = 'true'
    print("############################################################################")
    print("")
    print("")
    print("")
