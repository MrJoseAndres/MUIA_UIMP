import pandas as pd
import numpy as np
from skmultiflow.data import DataStream
from skmultiflow.evaluation import EvaluateHoldout
from skmultiflow.trees import HoeffdingTreeClassifier

#todo MODIFICAR PARA QUE IMPLEMENTE CORRECTAMENTE LA TÉNCICA DE EVALUACIÓN HOLDOUT

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

# Evaluation creation
evaluator = EvaluateHoldout(
    test_size=int(len(X) * 0.3),  # 30% for testing
    dynamic_test_set=False,
    max_samples=len(X),
    n_wait=250,
    show_plot=True,
    metrics=['accuracy', 'kappa']
)

# Evaluation
evaluator.evaluate(
    stream=stream,
    model=ht_classifier,
    model_names=['Hoeffding Tree']
)

# Paso 6: Imprimir resultados
print("\nEvaluación Completada")
print("----------------------------")
try:
    print(f"Precisión final del modelo: {evaluator.mean_eval_measurements[-1]['accuracy']:.4f}")
    print(f"Kappa final del modelo: {evaluator.mean_eval_measurements[-1]['kappa']:.4f}")
except (IndexError, TypeError):
    print("No se pudieron obtener las métricas finales.")

print(f"Muestras totales procesadas: {evaluator.global_sample_count}")

# Imprimir información adicional del conjunto de datos
print("\nInformación del conjunto de datos:")
print(f"Total de muestras: {n_samples}")
print(f"Muestras de entrenamiento: {n_samples - test_size}")
print(f"Muestras de test: {test_size}")