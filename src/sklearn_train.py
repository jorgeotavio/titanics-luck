import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from pathlib import Path
import os

def load_and_prepare_data(data_dir):
    train = pd.read_csv(data_dir + '/datasets/train.csv')
    test = pd.read_csv(data_dir + '/datasets/test.csv')

    train_ID = train.shape[0]

    target_train = train['Survived'].copy()
    train.drop(['Survived'], axis=1, inplace=True)

    all_data = pd.concat([train, test], axis=0).reset_index(drop=True)
    all_data = all_data[['Pclass', 'Sex', 'Age', 'SibSp', 'Fare', 'Parch', 'Embarked']].copy()

    all_data.fillna({'Age': all_data['Age'].median(), 'Fare': all_data['Fare'].median(), 'Embarked': 'S'}, inplace=True)

    label_encoder = LabelEncoder()
    all_data['Sex'] = label_encoder.fit_transform(all_data['Sex'])
    all_data['Embarked'] = label_encoder.fit_transform(all_data['Embarked'])

    train = all_data.iloc[:train_ID]
    test = all_data.iloc[train_ID:]

    return train, target_train, test

def evaluate_model(model, train, target_train, k=5):
    scores = cross_val_score(model, train, target_train, cv=k)
    return scores.mean()

def run_training():
    path = Path(os.getcwd())
    data_dir = str(path) + "/data/"

    train, target_train, test = load_and_prepare_data(data_dir)

    models = {
        'Regressão Logística': LogisticRegression(solver='liblinear'),
        'SVM': SVC(),
        'Árvore de Decisão': DecisionTreeClassifier()
    }

    for name, model in models.items():
        accuracy = evaluate_model(model, train, target_train)
        print(f"Acurácia média {name}: {accuracy * 100:.2f}%")
