import snowflake.connector
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
import datetime
import pandas as pd
import re
import html
import json
from tqdm import tqdm

# Snowflake connection parameters
conn = snowflake.connector.connect(
    user="TES_USR_LACAK",
    password="StrongPassword123",
    account="hb01677.ap-southeast-3.aws",
    database="TES_DB_RAW",
    schema="RAW_LACAK"
)

# Function to fetch data from Snowflake
def fetch_data_from_snowflake():
    print("Mengambil data dari Snowflake...")
    try:
        cursor = conn.cursor()
        query = "SELECT * FROM tr_track"
        cursor.execute(query)
        columns = [col[0] for col in cursor.description]
        results = cursor.fetchall()
        df = pd.DataFrame(results, columns=columns)
        print(f"Berhasil mengambil {len(df)} baris data dari Snowflake.")
        return df
    except Exception as e:
        print(f"Error saat mengambil data dari Snowflake: {e}")
        return None

# Convert duration (HH:MM:SS) to seconds
def duration_to_seconds(duration):
    if not duration:
        return 0
    try:
        h, m, s = map(int, duration.split(":"))
        return h * 3600 + m * 60 + s
    except:
        return 0

# Extract mileage from DESCRIPTION
def extract_mileage(description):
    if not description:
        return 0
    match = re.search(r"Mileage: (\d+\.?\d*)", str(description))
    return float(match.group(1)) if match else 0

# Preprocessing and feature engineering
def preprocess_data(data):
    print("Memproses data...")
    filtered_data = data[
        data["UNIT"].notnull() & 
        data["TYPE"].notnull() & 
        data["DATE_TIME_BEGIN"].notnull() & 
        data["DATE_TIME_END"].notnull()
    ]
    
    if filtered_data.empty:
        print("Tidak ada data yang valid setelah penyaringan.")
        return None, None, None, None
    
    print(f"Menemukan {len(filtered_data)} baris data untuk diproses.")
    processed_data = []
    total_duration = 0
    total_mileage = 0
    for _, row in tqdm(filtered_data.iterrows(), total=len(filtered_data), desc="Memproses baris data"):
        try:
            begin_date = row["DATE_TIME_BEGIN"].to_pydatetime() if isinstance(row["DATE_TIME_BEGIN"], pd.Timestamp) else datetime.datetime.strptime(row["DATE_TIME_BEGIN"], "%Y-%m-%d %H:%M:%S")
            end_date = row["DATE_TIME_END"].to_pydatetime() if isinstance(row["DATE_TIME_END"], pd.Timestamp) else datetime.datetime.strptime(row["DATE_TIME_END"], "%Y-%m-%d %H:%M:%S")
            
            duration = duration_to_seconds(row["DURATION"])
            mileage = extract_mileage(row["DESCRIPTION"])
            efficiency = mileage / (duration / 3600) if duration > 0 else 0  # km/h
            
            processed_row = {
                "UNIT": row["UNIT"],
                "TYPE": row["TYPE"],
                "INITIAL_LOCATION": row["INITIAL_LOCATION"],
                "FINAL_LOCATION": row["FINAL_LOCATION"],
                "DATE_TIME_BEGIN": begin_date,
                "DATE_TIME_END": end_date,
                "HOUR_OF_DAY": begin_date.hour,
                "DAY_OF_WEEK": begin_date.weekday(),
                "DURATION_SECONDS": duration,
                "MILEAGE": mileage,
                "EFFICIENCY": efficiency
            }
            processed_data.append(processed_row)
            total_duration += duration
            total_mileage += mileage
        except Exception as e:
            print(f"Error memproses baris {row}: {e}")
            continue
    
    le_type = LabelEncoder()
    le_location = LabelEncoder()
    print("Mengkodekan variabel kategorikal...")
    for row in tqdm(processed_data, desc="Mengkodekan label"):
        row["TYPE_ENCODED"] = le_type.fit_transform([row["TYPE"]])[0]
        row["INITIAL_LOCATION_ENCODED"] = le_location.fit_transform([row["INITIAL_LOCATION"]])[0]
    
    # Hitung statistik dasar
    avg_duration = total_duration / len(processed_data) if processed_data else 0
    avg_mileage = total_mileage / len(processed_data) if processed_data else 0
    stats = {
        "total_rows": len(processed_data),
        "avg_duration": round(avg_duration),
        "avg_mileage": "{:.2f}".format(avg_mileage)
    }
    
    print("Pemrosesan data selesai.")
    return processed_data, le_type, le_location, stats

# Apply K-Means clustering with Elbow Method
def apply_kmeans(data):
    if not data:
        print("Tidak ada data untuk clustering.")
        return None, None
    
    print("Menerapkan clustering K-Means...")
    features = []
    for row in tqdm(data, desc="Mengekstrak fitur"):
        features.append([
            row["DURATION_SECONDS"],
            row["MILEAGE"],
            row["EFFICIENCY"],
            row["INITIAL_LOCATION_ENCODED"]
        ])
    
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    
    # Elbow Method to find optimal number of clusters
    wcss = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, random_state=42)
        kmeans.fit(scaled_features)
        wcss.append(kmeans.inertia_)
    
    # Prepare WCSS data for Chart.js
    elbow_data = [
        {"clusters": i, "wcss": wcss[i-1]}
        for i in range(1, 11)
    ]
    
    # Choose optimal clusters (e.g., 4, adjust based on elbow curve)
    optimal_clusters = 4  # Adjust based on visual inspection or automated method
    kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
    clusters = kmeans.fit_predict(scaled_features)
    
    # Detect outliers (e.g., points far from cluster centroids)
    distances = kmeans.transform(scaled_features)
    max_distances = np.max(distances, axis=1)
    outlier_threshold = np.percentile(max_distances, 95)  # Top 5% as outliers
    outliers = max_distances > outlier_threshold
    
    for i, row in tqdm(enumerate(data), total=len(data), desc="Menetapkan label cluster"):
        row["CLUSTER"] = clusters[i]
        row["IS_OUTLIER"] = bool(outliers[i])
    
    print("Clustering K-Means selesai.")
    return data, elbow_data

# Generate simplified HTML report
def generate_html_report(data, le_type, le_location, stats, elbow_data):
    if not data:
        print("Tidak ada data untuk membuat laporan.")
        return None
    
    print("Membuat laporan HTML...")
    def clean_string(s):
        if s is None:
            return ""
        if not isinstance(s, str):
            s = str(s)
        s = "".join(c for c in s if c.isprintable())
        return html.escape(s, quote=True)
    
    # Aggregate data for visualizations and decision-making
    cluster_counts = {}
    cluster_details = {}
    location_counts = {}
    cluster_vehicles = {}  # Track vehicles per cluster for maintenance
    cluster_anomalies = {}  # Track anomaly details
    for row in tqdm(data, desc="Mengagregasi data"):
        cluster = row["CLUSTER"]
        if cluster not in cluster_details:
            cluster_details[cluster] = {"duration": [], "mileage": [], "efficiency": [], "locations": [], "outliers": 0}
            cluster_vehicles[cluster] = set()
            cluster_anomalies[cluster] = []
        cluster_counts[cluster] = cluster_counts.get(cluster, 0) + 1
        cluster_details[cluster]["duration"].append(row["DURATION_SECONDS"])
        cluster_details[cluster]["mileage"].append(row["MILEAGE"])
        cluster_details[cluster]["efficiency"].append(row["EFFICIENCY"])
        cluster_details[cluster]["locations"].append(row["INITIAL_LOCATION"])
        cluster_vehicles[cluster].add(row["UNIT"])
        if row["IS_OUTLIER"]:
            cluster_details[cluster]["outliers"] += 1
            cluster_anomalies[cluster].append({
                "unit": row["UNIT"],
                "duration": row["DURATION_SECONDS"],
                "mileage": row["MILEAGE"],
                "location": row["INITIAL_LOCATION"]
            })
        location_counts[row["INITIAL_LOCATION"]] = location_counts.get(row["INITIAL_LOCATION"], 0) + 1
    
    # Cluster summaries and labels
    cluster_summaries = []
    for cluster_id in cluster_details:
        details = cluster_details[cluster_id]
        avg_duration = np.mean(details["duration"])
        avg_mileage = np.mean(details["mileage"])
        avg_efficiency = np.mean(details["efficiency"])
        top_location = pd.Series(details["locations"]).mode()[0] if details["locations"] else "Tidak ada"
        
        # Label clusters based on characteristics
        if avg_duration > 3600 and avg_mileage > 50:
            cluster_label = "Perjalanan Jarak Jauh"
            insight = f"Perjalanan ini panjang (rata-rata {round(avg_duration/3600, 1)} jam, {avg_mileage:.1f} km), memerlukan perawatan kendaraan."
            action = f"Jadwalkan perawatan untuk {len(cluster_vehicles[cluster_id])} kendaraan: {', '.join(list(cluster_vehicles[cluster_id])[:3] or ['Tidak ada'])}..."
        elif avg_duration < 600 and avg_mileage < 10:
            cluster_label = "Parkir Singkat"
            insight = f"Berhenti singkat (rata-rata {round(avg_duration/60)} menit) menunjukkan aktivitas pengiriman/layanan."
            action = f"Tinjau jadwal untuk {len(cluster_vehicles[cluster_id])} kendaraan di {top_location} agar lebih efisien."
        elif avg_efficiency < 10:
            cluster_label = "Perjalanan Tidak Efisien"
            insight = f"Efisiensi rendah ({avg_efficiency:.1f} km/jam) menunjukkan kemacetan atau rute buruk di {top_location}."
            action = f"Gunakan rute alternatif berbasis lalu lintas untuk {len(cluster_vehicles[cluster_id])} kendaraan."
        else:
            cluster_label = "Perjalanan Reguler"
            insight = f"Perjalanan seimbang (durasi {round(avg_duration/60)} menit, {avg_mileage:.1f} km) di {top_location}."
            action = f"Pantau {len(cluster_vehicles[cluster_id])} kendaraan untuk penggunaan bahan bakar optimal."
        
        cluster_summaries.append({
            "cluster": int(cluster_id),  # Ensure Python int
            "label": cluster_label,
            "count": int(cluster_counts.get(cluster_id, 0)),  # Convert to Python int
            "avg_duration": round(avg_duration / 60),  # Convert to minutes
            "avg_mileage": "{:.2f}".format(avg_mileage),
            "avg_efficiency": "{:.2f}".format(avg_efficiency),
            "top_location": clean_string(top_location),
            "outliers": int(details["outliers"]),  # Convert to Python int
            "vehicles": list(cluster_vehicles[cluster_id])[:3],  # Limit to 3 for display
            "insight": insight,
            "action": action
        })
    
    # Top locations for route optimization
    top_locations = [
        {"location": clean_string(k), "count": int(v)}  # Convert count to Python int
        for k, v in sorted(location_counts.items(), key=lambda x: x[1], reverse=True)[:3]
    ]
    
    # Cluster chart data
    cluster_chart_data = [
        {"cluster": f"Kelompok {s['cluster']} ({s['label']})", "count": s['count']}
        for s in cluster_summaries
    ]
    
    # Anomaly details for detection
    anomaly_details = []
    for cluster_id, anomalies in cluster_anomalies.items():
        for anomaly in anomalies[:3]:  # Limit to 3 per cluster for brevity
            anomaly_details.append({
                "cluster": int(cluster_id),
                "unit": clean_string(anomaly["unit"]),
                "duration": round(anomaly["duration"] / 60),
                "mileage": "{:.2f}".format(anomaly["mileage"]),
                "location": clean_string(anomaly["location"])
            })
    
    try:
        # Custom default function to handle NumPy types
        def json_default(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
        
        cluster_chart_json = json.dumps(cluster_chart_data, ensure_ascii=False, default=json_default)
        location_chart_json = json.dumps(top_locations, ensure_ascii=False, default=json_default)
        cluster_summaries_json = json.dumps(cluster_summaries, ensure_ascii=False, default=json_default)
        elbow_data_json = json.dumps(elbow_data, ensure_ascii=False, default=json_default)
        anomaly_details_json = json.dumps(anomaly_details, ensure_ascii=False, default=json_default)
    except Exception as e:
        print(f"Error saat membuat JSON: {e}")
        return None
    
    html_content = f"""
    <!DOCTYPE html>
    <html lang="id">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Laporan Pelacakan Kendaraan</title>
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&display=swap" rel="stylesheet">
        <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
        <script src="https://cdn.tailwindcss.com"></script>
        <style>
            body {{
                font-family: 'Inter', sans-serif;
                background: #f7fafc;
                margin: 0;
                padding: 24px;
            }}
            .container {{
                max-width: 900px;
                margin: 0 auto;
                background: white;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
                padding: 24px;
            }}
            .chart {{
                height: 200px !important;
            }}
            .tooltip {{
                position: relative;
                display: inline-block;
                cursor: pointer;
            }}
            .tooltip .tooltiptext {{
                visibility: hidden;
                width: 250px;
                background-color: #4a5568;
                color: white;
                text-align: center;
                border-radius: 4px;
                padding: 8px;
                position: absolute;
                z-index: 1;
                bottom: 125%;
                left: 50%;
                margin-left: -125px;
                opacity: 0;
                transition: opacity 0.3s;
            }}
            .tooltip:hover .tooltiptext {{
                visibility: visible;
                opacity: 1;
            }}
            .section {{
                border-left: 4px solid #4a90e2;
                padding-left: 16px;
                margin-bottom: 24px;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1 class="text-2xl font-semibold text-gray-800 mb-4">Laporan Pelacakan Kendaraan</h1>
            <p class="text-gray-600 mb-6">
                Laporan ini mengelompokkan data pergerakan kendaraan untuk membantu Anda mengambil keputusan penting seperti menghemat bahan bakar, menjaga kondisi kendaraan, menemukan masalah, dan meningkatkan efisiensi operasional.
            </p>

            <div class="section">
                <h2 class="text-lg font-semibold text-gray-700 mb-2">Ringkasan Data</h2>
                <p class="text-gray-600">
                    <strong>Jumlah Peristiwa:</strong> {stats['total_rows']}<br>
                    <strong>Rata-rata Durasi:</strong> {stats['avg_duration']} detik<br>
                    <strong>Rata-rata Jarak:</strong> {stats['avg_mileage']} km
                </p>
            </div>

            <div class="section">
                <h2 class="text-lg font-semibold text-gray-700 mb-2">1. Optimasi Rute</h2>
                <p class="text-gray-600 mb-4">
                    Bagian ini membantu mengurangi waktu dan bahan bakar dengan memilih jalur terbaik berdasarkan lokasi yang sering dikunjungi dan efisiensi perjalanan.
                </p>
                <canvas id="location-chart" class="chart mb-4"></canvas>
                <ul class="list-disc pl-5 text-gray-600">
                    {"".join(f'<li><strong>{loc["location"]}</strong>: Dikunjungi {loc["count"]} kali. Pertimbangkan jalur alternatif untuk mengurangi kemacetan.</li>' for loc in top_locations)}
                    {"".join(f'<li><strong>Kelompok {s["cluster"]} ({s["label"]})</strong>: Efisiensi {s["avg_efficiency"]} km/jam. {s["action"]}</li>' for s in cluster_summaries if s["label"] == "Perjalanan Tidak Efisien")}
                </ul>
            </div>

            <div class="section">
                <h2 class="text-lg font-semibold text-gray-700 mb-2">2. Perencanaan Perawatan</h2>
                <p class="text-gray-600 mb-4">
                    Bagian ini mengidentifikasi kendaraan yang sering melakukan perjalanan jauh untuk memastikan perawatan tepat waktu, mencegah kerusakan.
                </p>
                <canvas id="cluster-chart" class="chart mb-4"></canvas>
                <ul class="list-disc pl-5 text-gray-600">
                    {"".join(f'<li><strong>Kelompok {s["cluster"]} ({s["label"]})</strong>: {s["insight"]} <span class="tooltip">📋<span class="tooltiptext">{s["action"]}</span></span></li>' for s in cluster_summaries if s["label"] == "Perjalanan Jarak Jauh")}
                </ul>
            </div>

            <div class="section">
                <h2 class="text-lg font-semibold text-gray-700 mb-2">3. Deteksi Anomali</h2>
                <p class="text-gray-600 mb-4">
                    Bagian ini menemukan perjalanan atau parkir yang tidak biasa (anomali) untuk ditindaklanjuti, seperti kesalahan pengemudi atau masalah kendaraan.
                </p>
                <ul class="list-disc pl-5 text-gray-600">
                    {"".join(f'<li>Kendaraan <strong>{a["unit"]}</strong> di Kelompok {a["cluster"]}: Durasi {a["duration"]} menit, Jarak {a["mileage"]} km di {a["location"]}. Periksa apakah ini normal.</li>' for a in anomaly_details)}
                </ul>
            </div>

            <div class="section">
                <h2 class="text-lg font-semibold text-gray-700 mb-2">4. Peningkatan Efisiensi</h2>
                <p class="text-gray-600 mb-4">
                    Bagian ini membantu mengurangi waktu parkir atau perjalanan tidak efisien untuk menghemat biaya operasional.
                </p>
                <ul class="list-disc pl-5 text-gray-600">
                    {"".join(f'<li><strong>Kelompok {s["cluster"]} ({s["label"]})</strong>: {s["insight"]} <span class="tooltip">📋<span class="tooltiptext">{s["action"]}</span></span></li>' for s in cluster_summaries if s["label"] in ["Parkir Singkat", "Perjalanan Tidak Efisien"])}
                </ul>
            </div>

            <div class="section">
                <h2 class="text-lg font-semibold text-gray-700 mb-2">Kurva Elbow (Jumlah Kelompok Optimal)</h2>
                <p class="text-gray-600 mb-4">
                    Grafik ini menunjukkan jumlah kelompok yang ideal untuk mengelompokkan data kendaraan.
                </p>
                <canvas id="elbow-chart" class="chart"></canvas>
            </div>

            <div class="section">
                <h2 class="text-lg font-semibold text-gray-700 mb-2">Ringkasan Kelompok</h2>
                <table class="w-full text-left border-collapse">
                    <thead>
                        <tr class="bg-gray-100">
                            <th class="p-2">Kelompok</th>
                            <th class="p-2">Jumlah</th>
                            <th class="p-2">Durasi (menit)</th>
                            <th class="p-2">Jarak (km)</th>
                            <th class="p-2">Efisiensi (km/jam)</th>
                            <th class="p-2">Lokasi Utama</th>
                            <th class="p-2">Anomali</th>
                        </tr>
                    </thead>
                    <tbody>
                        {"".join(f'<tr><td class="p-2">{s["label"]}</td><td class="p-2">{s["count"]}</td><td class="p-2">{s["avg_duration"]}</td><td class="p-2">{s["avg_mileage"]}</td><td class="p-2">{s["avg_efficiency"]}</td><td class="p-2">{s["top_location"]}</td><td class="p-2">{s["outliers"]}</td></tr>' for s in cluster_summaries)}
                    </tbody>
                </table>
            </div>
        </div>

        <script>
            const locationChartData = {location_chart_json};
            const clusterChartData = {cluster_chart_json};
            const elbowData = {elbow_data_json};
            const anomalyDetails = {anomaly_details_json};

            function createBarChart(canvasId, data, labelKey, color) {{
                const ctx = document.getElementById(canvasId).getContext('2d');
                new Chart(ctx, {{
                    type: 'bar',
                    data: {{
                        labels: data.map(item => item[labelKey].length > 15 ? item[labelKey].substring(0, 12) + '...' : item[labelKey]),
                        datasets: [{{
                            label: 'Jumlah',
                            data: data.map(item => item.count),
                            backgroundColor: color,
                            borderColor: color,
                            borderWidth: 1
                        }}]
                    }},
                    options: {{
                        responsive: true,
                        maintainAspectRatio: false,
                        scales: {{
                            y: {{ beginAtZero: true, title: {{ display: true, text: 'Jumlah' }} }},
                            x: {{ title: {{ display: true, text: labelKey.charAt(0).toUpperCase() + labelKey.slice(1) }} }}
                        }},
                        plugins: {{
                            legend: {{ display: false }},
                            tooltip: {{ enabled: true }}
                        }}
                    }}
                }});
            }}

            function createElbowChart(canvasId, data) {{
                const ctx = document.getElementById(canvasId).getContext('2d');
                new Chart(ctx, {{
                    type: 'line',
                    data: {{
                        labels: data.map(item => item.clusters),
                        datasets: [{{
                            label: 'WCSS',
                            data: data.map(item => item.wcss),
                            borderColor: '#38a169',
                            backgroundColor: 'rgba(56, 161, 105, 0.1)',
                            fill: true,
                            tension: 0.4
                        }}]
                    }},
                    options: {{
                        responsive: true,
                        maintainAspectRatio: false,
                        scales: {{
                            y: {{ beginAtZero: false, title: {{ display: true, text: 'WCSS' }} }},
                            x: {{ title: {{ display: true, text: 'Jumlah Kelompok' }} }}
                        }},
                        plugins: {{
                            legend: {{ display: true }},
                            tooltip: {{ enabled: true }}
                        }}
                    }}
                }});
            }}

            createBarChart('location-chart', locationChartData, 'location', '#4a90e2');
            createBarChart('cluster-chart', clusterChartData, 'cluster', '#e53e3e');
            createElbowChart('elbow-chart', elbowData);
        </script>
    </body>
    </html>
    """
    
    with open("vehicle_tracking_report.html", "w", encoding="utf-8") as f:
        f.write(html_content)
    print("Pembuatan laporan HTML selesai.")
    return html_content

# Main execution
def main():
    print("Memulai analisis pelacakan kendaraan...")
    
    # Langkah 1: Fetch data
    data = fetch_data_from_snowflake()
    if data is None:
        print("Gagal mengambil data. Keluar.")
        return
    
    # Langkah 2: Preprocess data
    processed_data, le_type, le_location, stats = preprocess_data(data)
    if processed_data is None:
        print("Gagal memproses data. Keluar.")
        return
    
    # Langkah 3: Apply clustering
    clustered_data, elbow_data = apply_kmeans(processed_data)
    if clustered_data is None:
        print("Gagal menerapkan clustering. Keluar.")
        return
    
    # Langkah 4: Generate report
    generate_html_report(clustered_data, le_type, le_location, stats, elbow_data)
    print("Analisis pelacakan kendaraan selesai.")

if __name__ == "__main__":
    main()