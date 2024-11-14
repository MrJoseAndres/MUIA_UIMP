import pandas as pd
import numpy as np
from skmultiflow.data import DataStream
from skmultiflow.evaluation import EvaluatePrequential
from skmultiflow.trees import HoeffdingTreeClassifier
from skmultiflow.bayes import NaiveBayes

# Loading data
data = pd.read_csv('./data/adult.csv', delimiter=',')

# Splitting data into X and y
X = data.drop(columns=['income'])
y = pd.DataFrame(data['income'])

# For each colums, if it is an object, we will convert it to a number
nominal_attributes = []
val_to_embed = {}
for col in X.columns:    
    if X[col].dtype == 'object':
        val_to_embed[col] = {val: i for i, val in enumerate(X[col].unique())}
        X[col] = X[col].apply(lambda x: val_to_embed[col][x])
        nominal_attributes.append(col)

val_to_embed["income"] = {val: i for i, val in enumerate(y["income"].unique())}
y["income"] = y["income"].apply(lambda x: val_to_embed["income"][x])
nominal_attributes.append('income')

# Data Stream cration
stream = DataStream(X, y)

# Model creation
ht_classifier = HoeffdingTreeClassifier(
    nominal_attributes=nominal_attributes,
    leaf_prediction='nb'
)

nb_classifier = NaiveBayes()

# Evaluation creation
n_samples = len(X)
pretraining_size = int(n_samples * 0.05)
evaluator = EvaluatePrequential(
    max_samples=n_samples,
    pretrain_size=pretraining_size,
    show_plot=True,
    metrics=['accuracy', 'kappa']
)

# Evaluation
evaluator.evaluate(
    stream=stream,
    model=[ht_classifier,nb_classifier],
    model_names=['Hoeffding Tree', 'Naive Bayes']
)

test_size = int(n_samples)
print("\nInformación del conjunto de datos:")
print(f"Total de muestras: {n_samples}")
print(f"Muestras de entrenamiento: {n_samples - test_size}")
print(f"Muestras de test: {test_size}")