import pandas as pd
from skmultiflow.data import DataStream
from skmultiflow.trees import HoeffdingTreeClassifier
from skmultiflow.bayes import NaiveBayes
from skmultiflow.drift_detection.adwin import ADWIN

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

# Inicializar modelos
ht_classifier = HoeffdingTreeClassifier()
nb_classifier = NaiveBayes()

# Inicializar ADWIN para detección de deriva
adwin_ht = ADWIN()
adwin_nb = ADWIN()

# Variables para registrar drifts
detected_drifts_ht = []
detected_drifts_nb = []

# Evaluar el flujo de datos
n_processed = 0
while stream.has_more_samples():
    X_batch, y_batch = stream.next_sample()
    
    # Test
    pred_ht = ht_classifier.predict(X_batch)
    pred_nb = nb_classifier.predict(X_batch)

    # Check drift
    adwin_ht.add_element(int(pred_ht != y_batch))
    adwin_nb.add_element(int(pred_nb != y_batch))
    
    if adwin_ht.detected_change():
        print(f"[ADWIN] Drift detectado en Hoeffding Tree en muestra {n_processed}")
        detected_drifts_ht.append(n_processed)

    if adwin_nb.detected_change():
        print(f"[ADWIN] Drift detectado en Naive Bayes en muestra {n_processed}")
        detected_drifts_nb.append(n_processed)
    
    # Then Train
    ht_classifier.partial_fit(X_batch, y_batch)
    nb_classifier.partial_fit(X_batch, y_batch)

    n_processed += 1

# Prints
print("\nResumen:")
print(f"Muestras procesadas: {n_processed}")
print(f"Drifts detectados Hoeffding Tree: {len(detected_drifts_ht)} en muestras {detected_drifts_ht}")
print(f"Drifts detectados Naive Bayes: {len(detected_drifts_nb)} en muestras {detected_drifts_nb}")