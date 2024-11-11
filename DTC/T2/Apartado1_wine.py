import pandas as pd
import numpy as np
from skmultiflow.data import DataStream
from skmultiflow.evaluation import EvaluateHoldout
from skmultiflow.trees import HoeffdingTreeClassifier

# Paso 1: Cargar y preparar los datos
data = pd.read_csv('./winequality-white.csv', delimiter=';')

# Asegurarnos de que los datos están en el formato correcto
X = data.drop(columns=['quality']).values.astype(np.float64)
y = data['quality'].values.astype(np.int32)

# Calcular el tamaño del conjunto de test
n_samples = len(X)
test_size = int(n_samples * 0.3)  # 30% para test

# Paso 2: Crear el stream de datos
stream = DataStream(X, y)

# Paso 3: Configurar el clasificador
ht_classifier = HoeffdingTreeClassifier(
    nominal_attributes=None,
    leaf_prediction='nb'
)

# Paso 4: Configurar la evaluación
evaluator = EvaluateHoldout(
    test_size=test_size,  # Usar el número exacto de muestras
    dynamic_test_set=False,
    max_samples=n_samples,
    n_wait=250,
    show_plot=True,
    metrics=['accuracy', 'kappa']
)

# Paso 5: Ejecutar la evaluación
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

# Imprimir detalles del árbol
try:
    print("\nDetalles del clasificador:")
    print(f"Altura del árbol: {ht_classifier.get_model_measurements()[1]}")
    print(f"Número de nodos: {ht_classifier.get_model_measurements()[0]}")
except:
    print("No se pudieron obtener las métricas del árbol.")

# Imprimir información adicional del conjunto de datos
print("\nInformación del conjunto de datos:")
print(f"Total de muestras: {n_samples}")
print(f"Muestras de entrenamiento: {n_samples - test_size}")
print(f"Muestras de test: {test_size}")