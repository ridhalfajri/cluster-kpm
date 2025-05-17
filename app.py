from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
import numpy as np

app = Flask(__name__)
CORS(app)

@app.route('/cluster', methods=['POST'])
def cluster_data():
    data = request.json
    print(data)
    if not data:
        return jsonify({'status': 'error', 'message': 'No data received'}), 400
    df = pd.DataFrame(data['data'])
    if 'bobot' not in data:
        return jsonify({'status': 'error', 'message': "'bobot' field is required"}), 400
    bobot = data.get('bobot')

    for feature, weight in bobot.items():
        df[feature] = df[feature] * weight  # Bobot ditambahkan pada masing-masing fitur

    # Fitur yang digunakan untuk klasterisasi
    X = df[['working_st', 'disability_st', 'chronic_disease_type', 'single_elderly_family']]

    # Klasterisasi Menggunakan KMeans
    kmeans = KMeans(n_clusters=3, init='k-means++', random_state=42)
    df['cluster'] = kmeans.fit_predict(X)

    # Menyusun Hasil Klaster dalam Format JSON untuk response
    cluster_result = {}
    for cluster_id in range(3):  # Asumsi 3 cluster
        cluster_result[f'cluster {cluster_id}'] = df[df['cluster'] == cluster_id].to_dict(orient='records')

    # Proses untuk Memenuhi Kuota BLT
    kuota = data['kuota']  # Mendapatkan kuota dari request JSON
    cluster_sizes = df['cluster'].value_counts().to_dict()

    # Menghitung total bobot per cluster
    cluster_bobot = {}
    for cluster_id in cluster_sizes:
        cluster_data = df[df['cluster'] == cluster_id]
        total_bobot = cluster_data[['working_st', 'disability_st', 'chronic_disease_type', 'single_elderly_family']].sum().sum()
        cluster_bobot[cluster_id] = total_bobot

    # Mengurutkan cluster berdasarkan total bobot (cluster dengan bobot tertinggi dulu)
    sorted_clusters = sorted(cluster_bobot.items(), key=lambda x: x[1], reverse=True)

    selected_individuals = []
    remaining_kuota = kuota

    # Pilih individu berdasarkan cluster dengan bobot tertinggi
    for cluster_id, _ in sorted_clusters:
        if remaining_kuota <= 0:
            break

        cluster_size = cluster_sizes.get(cluster_id, 0)
        if cluster_size >= remaining_kuota:
            selected_individuals.extend(select_with_decision_tree(df[df['cluster'] == cluster_id], remaining_kuota))
            remaining_kuota = 0
        else:
            selected_individuals.extend(df[df['cluster'] == cluster_id].to_dict(orient='records'))
            remaining_kuota -= cluster_size

    # Mengembalikan hasil clustering dan penerima BLT yang dipilih
    return jsonify({
        'status': 'success',
        'message': 'Clustering completed and BLT recipients selected',
        # 'cluster_result': cluster_result,  # Menambahkan hasil cluster
        'data': selected_individuals
    })


def select_with_decision_tree(df, kuota):
    features = ['working_st', 'disability_st', 'chronic_disease_type', 'single_elderly_family']
    X = df[features]

    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X, np.zeros(len(X)))  # For decision tree, we use dummy labels (0)

    # Using .loc to avoid the SettingWithCopyWarning
    df.loc[:, 'priority'] = clf.predict_proba(X)[:, 0]

    df_sorted = df.sort_values(by='priority', ascending=False)

    selected_individuals = df_sorted.head(kuota).to_dict(orient='records')
    return selected_individuals

if __name__ == '__main__':
    app.run(debug=True)
