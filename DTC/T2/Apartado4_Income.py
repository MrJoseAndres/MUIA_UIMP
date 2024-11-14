import pandas as pd
import numpy as np
from sklearn.cluster import Birch
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from river import cluster

# Loading data
data = pd.read_csv('./data/adult.csv', delimiter=',')

# Removing columns that are not useful because we cannot measure distance between them
data = data[['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']]

# Data normalization
data = (data - data.mean()) / data.std()

# BIRCH initialization
birch_model = Birch(n_clusters=None) #? threshold=0.5
dbstream_model = cluster.DBSTREAM()

# Scan all data
iterations = 0
for row in data.iterrows():
    sample = row[1].values.reshape(1, -1)  # Convertir fila a array 2D

    birch_model.partial_fit(sample)
    x_dict = {f'feature_{i}': sample[0][i] for i in range(sample.shape[1])}
    dbstream_model = dbstream_model.learn_one(x_dict)
    iterations += 1
    if iterations % 1000 == 0:
        print(f"Processed {round(iterations/len(data) * 100, 2)}% of the samples")
print(f"All samples processed")

# Global clustering
print("Global clustering...")
birch_cluster_assignments = birch_model.predict(data.values.reshape(-1, 6))

data_dicts = [{f'feature_{i}': row[i] for i in range(len(row))} for row in data.values]
dbstream_assignments = [dbstream_model.predict_one(data_dict) for data_dict in data_dicts]

# Metrics calculation
metrics = {}
print("Calculating metrics...")
metrics['BIRCH'] = {
    'n_clusters': len(np.unique(birch_cluster_assignments)),
    'silhouette': silhouette_score(data.values, birch_cluster_assignments),
    'calinski': calinski_harabasz_score(data.values, birch_cluster_assignments),
    'davies': davies_bouldin_score(data.values, birch_cluster_assignments)
}
metrics['DBSTREAM'] = {
    'n_clusters': len(np.unique(dbstream_assignments)),
    'silhouette': silhouette_score(data.values, dbstream_assignments),
    'calinski': calinski_harabasz_score(data.values, dbstream_assignments),
    'davies': davies_bouldin_score(data.values, dbstream_assignments)
}

models = ['BIRCH', 'DBSTREAM']

for model in models:
    print("="*50)
    print(f"Model: {model}")
    print("-"*50)
    print(f"Number of clusters: {metrics[model]['n_clusters']}")
    print(f"Silhouette score: {metrics[model]['silhouette']}")
    print(f"Calinski-Harabasz score: {metrics[model]['calinski']}")
    print(f"Davies-Bouldin score: {metrics[model]['davies']}")
print("="*50)