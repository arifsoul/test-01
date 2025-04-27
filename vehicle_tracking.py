import snowflake.connector
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler, LabelEncoder
import datetime
import pandas as pd
import re
import html
import json
from tqdm import tqdm  # Import tqdm untuk progress bar

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
    print("Fetching data from Snowflake...")
    try:
        cursor = conn.cursor()
        query = "SELECT * FROM tr_track"
        cursor.execute(query)
        columns = [col[0] for col in cursor.description]
        results = cursor.fetchall()
        df = pd.DataFrame(results, columns=columns)
        print(f"Successfully fetched {len(df)} rows from Snowflake.")
        return df
    except Exception as e:
        print(f"Error connecting to Snowflake: {e}")
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
    print("Preprocessing data...")
    filtered_data = data[
        data["UNIT"].notnull() & 
        data["TYPE"].notnull() & 
        data["DATE_TIME_BEGIN"].notnull() & 
        data["DATE_TIME_END"].notnull()
    ]
    
    if filtered_data.empty:
        print("No valid data after filtering.")
        return None, None, None
    
    print(f"Filtered {len(filtered_data)} rows for processing.")
    processed_data = []
    # Gunakan tqdm untuk menampilkan progress bar saat memproses baris
    for _, row in tqdm(filtered_data.iterrows(), total=len(filtered_data), desc="Processing rows"):
        try:
            begin_date = row["DATE_TIME_BEGIN"].to_pydatetime() if isinstance(row["DATE_TIME_BEGIN"], pd.Timestamp) else datetime.datetime.strptime(row["DATE_TIME_BEGIN"], "%Y-%m-%d %H:%M:%S")
            end_date = row["DATE_TIME_END"].to_pydatetime() if isinstance(row["DATE_TIME_END"], pd.Timestamp) else datetime.datetime.strptime(row["DATE_TIME_END"], "%Y-%m-%d %H:%M:%S")
            
            processed_row = {
                "UNIT": row["UNIT"],
                "TYPE": row["TYPE"],
                "INITIAL_LOCATION": row["INITIAL_LOCATION"],
                "FINAL_LOCATION": row["FINAL_LOCATION"],
                "DATE_TIME_BEGIN": begin_date,
                "DATE_TIME_END": end_date,
                "HOUR_OF_DAY": begin_date.hour,
                "DAY_OF_WEEK": begin_date.weekday(),
                "DURATION_SECONDS": duration_to_seconds(row["DURATION"]),
                "MILEAGE": extract_mileage(row["DESCRIPTION"])
            }
            processed_data.append(processed_row)
        except Exception as e:
            print(f"Error processing row {row}: {e}")
            continue
    
    types = [row["TYPE"] for row in processed_data]
    locations = [row["INITIAL_LOCATION"] for row in processed_data]
    le_type = LabelEncoder()
    le_location = LabelEncoder()
    # Gunakan tqdm untuk encoding
    print("Encoding categorical variables...")
    for row in tqdm(processed_data, desc="Encoding labels"):
        row["TYPE_ENCODED"] = le_type.fit_transform([row["TYPE"]])[0]
        row["INITIAL_LOCATION_ENCODED"] = le_location.fit_transform([row["INITIAL_LOCATION"]])[0]
    
    print("Preprocessing completed.")
    return processed_data, le_type, le_location

# Apply DBSCAN clustering
def apply_dbscan(data):
    if not data:
        print("No data for clustering.")
        return None
    
    print(f"Applying DBSCAN clustering to {len(data)} data points...")
    features = []
    # Gunakan tqdm untuk membuat fitur
    for row in tqdm(data, desc="Extracting features"):
        features.append([
            row["DURATION_SECONDS"], row["MILEAGE"], row["HOUR_OF_DAY"], row["DAY_OF_WEEK"],
            row["TYPE_ENCODED"], row["INITIAL_LOCATION_ENCODED"]
        ])
    
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    clusters = dbscan.fit_predict(scaled_features)
    
    # Gunakan tqdm untuk menambahkan label cluster
    for i, row in tqdm(enumerate(data), total=len(data), desc="Assigning clusters"):
        row["CLUSTER"] = clusters[i]
    
    print("DBSCAN clustering completed.")
    return data

# Generate HTML report with Chart.js
def generate_html_report(data, le_type, le_location, stats):
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
    
    # Aggregate data for visualizations
    type_counts = {}
    location_counts = {}
    cluster_counts = {}
    cluster_details = {i: {"duration": [], "mileage": [], "types": [], "locations": []} for i in range(5)}
    
    for row in tqdm(data, desc="Mengagregasi data"):
        type_counts[row["TYPE"]] = type_counts.get(row["TYPE"], 0) + 1
        location_counts[row["INITIAL_LOCATION"]] = location_counts.get(row["INITIAL_LOCATION"], 0) + 1
        cluster_counts[row["CLUSTER"]] = cluster_counts.get(row["CLUSTER"], 0) + 1
        cluster = row["CLUSTER"]
        cluster_details[cluster]["duration"].append(row["DURATION_SECONDS"])
        cluster_details[cluster]["mileage"].append(row["MILEAGE"])
        cluster_details[cluster]["types"].append(row["TYPE"])
        cluster_details[cluster]["locations"].append(row["INITIAL_LOCATION"])
    
    # Batasi cluster ke top 3 dan kelompokkan sisanya ke "Lainnya"
    top_clusters = sorted(cluster_counts.items(), key=lambda x: x[1], reverse=True)[:3]
    other_count = sum(v for k, v in cluster_counts.items() if k not in [x[0] for x in top_clusters])
    if other_count > 0:
        top_clusters.append((-1, other_count))
    
    type_chart_data = [{"type": clean_string(k), "count": v} for k, v in type_counts.items()]
    location_chart_data = [
        {"location": clean_string(k), "count": v} for k, v in
        sorted(location_counts.items(), key=lambda x: x[1], reverse=True)[:3]
    ]
    cluster_chart_data = [
        {"cluster": clean_string("Lainnya" if k == -1 else str(k)), "count": v} for k, v in top_clusters
    ]
    
    # Hitung detail statistik untuk setiap cluster
    cluster_stats = []
    for cluster_id in range(5):
        details = cluster_details[cluster_id]
        if not details["duration"]:
            continue
        avg_duration = np.mean(details["duration"])
        avg_mileage = np.mean(details["mileage"])
        most_common_type = pd.Series(details["types"]).mode()[0] if details["types"] else "Tidak ada"
        most_common_location = pd.Series(details["locations"]).mode()[0] if details["locations"] else "Tidak ada"
        cluster_stats.append({
            "cluster": cluster_id,
            "count": cluster_counts.get(cluster_id, 0),
            "avg_duration": round(avg_duration),
            "avg_mileage": "{:.2f}".format(avg_mileage),
            "most_common_type": clean_string(most_common_type),
            "most_common_location": clean_string(most_common_location)
        })
    
    # Generate HTML using f-string
    html_content = f"""
    <script type="text/javascript">
        var gk_isXlsx = false;
        var gk_xlsxFileLookup = {{}};
        var gk_fileData = {{}};
        function filledCell(cell) {{
          return cell !== '' && cell != null;
        }}
        function loadFileData(filename) {{
            if (gk_isXlsx && gk_xlsxFileLookup[filename]) {{
                try {{
                    var workbook = XLSX.read(gk_fileData[filename], {{ type: 'base64' }});
                    var firstSheetName = workbook.SheetNames[0];
                    var worksheet = workbook.Sheets[firstSheetName];
                    var jsonData = XLSX.utils.sheet_to_json(worksheet, {{ header: 1, blankrows: false, defval: '' }});
                    var filteredData = jsonData.filter(row => row.some(filledCell));
                    var headerRowIndex = filteredData.findIndex((row, index) =>
                      row.filter(filledCell).length >= filteredData[index + 1]?.filter(filledCell).length
                    );
                    if (headerRowIndex === -1 || headerRowIndex > 25) {{
                      headerRowIndex = 0;
                    }}
                    var csv = XLSX.utils.aoa_to_sheet(filteredData.slice(headerRowIndex));
                    csv = XLSX.utils.sheet_to_csv(csv, {{ header: 1 }});
                    return csv;
                }} catch (e) {{
                    console.error(e);
                    return "";
                }}
            }}
            return gk_fileData[filename] || "";
        }}
    </script>
    <!DOCTYPE html>
    <html lang="id">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Laporan Analisis Pelacakan Kendaraan</title>
        <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap" rel="stylesheet">
        <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
        <script src="https://cdn.jsdelivr.net/npm/chartjs-plugin-datalabels@2.0.0/dist/chartjs-plugin-datalabels.min.js"></script>
        <script src="https://cdn.tailwindcss.com"></script>
        <style>
            body {{
                font-family: 'Poppins', sans-serif;
                background: linear-gradient(135deg, #f0f4f8 0%, #d9e2ec 100%);
                margin: 0;
                padding: 24px;
                font-size: 16px;
            }}
            .chart-container {{
                margin: 12px 0;
                max-width: 600px;
                background: white;
                border-radius: 12px;
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
                padding: 16px;
            }}
            .error {{
                color: #ef4444;
                font-weight: 600;
            }}
            .chart {{
                width: 100%;
                height: 300px !important;
            }}
            .section-title {{
                color: #1f2937;
                border-bottom: 3px solid #3b82f6;
                padding-bottom: 6px;
            }}
            .report-container {{
                background: white;
                border-radius: 16px;
                box-shadow: 0 8px 16px rgba(0, 0, 0, 0.15);
                padding: 24px;
                max-width: 900px;
                margin: 0 auto;
            }}
        </style>
    </head>
    <body>
        <div id="root" class="report-container"></div>
        <div id="error" class="error" style="display: none;"></div>
        <script>
            Chart.register(ChartDataLabels);
            const typeChartData = [{','.join(f'{{ "type": "{item["type"]}", "count": {item["count"]} }}' for item in type_chart_data) if type_chart_data else ''}];
            const locationChartData = [{','.join(f'{{ "location": "{item["location"]}", "count": {item["count"]} }}' for item in location_chart_data) if location_chart_data else ''}];
            const clusterChartData = [{','.join(f'{{ "cluster": "{item["cluster"]}", "count": {item["count"]} }}' for item in cluster_chart_data) if cluster_chart_data else ''}];
            const clusterStats = [{','.join(f'{{ "cluster": {item["cluster"]}, "count": {item["count"]}, "avg_duration": {item["avg_duration"]}, "avg_mileage": "{item["avg_mileage"]}", "most_common_type": "{item["most_common_type"]}", "most_common_location": "{item["most_common_location"]}" }}' for item in cluster_stats) if cluster_stats else ''}];
            const stats = {{ total_rows: {stats['total_rows']}, avg_duration: {stats['avg_duration']}, avg_mileage: "{stats['avg_mileage']}" }};

            function showError(message) {{
                const errorDiv = document.getElementById('error');
                errorDiv.style.display = 'block';
                errorDiv.textContent = message;
            }}

            function formatNumber(value) {{
                if (value >= 1000000) return (value / 1000000) + 'M';
                if (value >= 1000) return (value / 1000) + 'K';
                return value;
            }}

            function createBarChart(canvasId, data, labelKey, color) {{
                try {{
                    const truncatedLabels = data.map(item => {{
                        const label = item[labelKey];
                        return label.length > 10 ? label.substring(0, 8) + '...' : label;
                    }});
                    const ctx = document.getElementById(canvasId).getContext('2d');
                    new Chart(ctx, {{
                        type: 'bar',
                        data: {{
                            labels: truncatedLabels,
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
                            animation: {{
                                duration: 1000,
                                easing: 'easeOutQuart'
                            }},
                            scales: {{
                                y: {{
                                    beginAtZero: true,
                                    max: 100000,
                                    title: {{ display: true, text: 'Jumlah', font: {{ size: 14, family: 'Poppins' }} }},
                                    ticks: {{
                                        font: {{ size: 12, family: 'Poppins' }},
                                        callback: function(value) {{
                                            return formatNumber(value);
                                        }}
                                    }}
                                }},
                                x: {{
                                    title: {{ display: true, text: labelKey.charAt(0).toUpperCase() + labelKey.slice(1), font: {{ size: 14, family: 'Poppins' }} }},
                                    ticks: {{
                                        font: {{ size: 12, family: 'Poppins' }},
                                        autoSkip: true,
                                        maxTicksLimit: 4,
                                        maxRotation: 0,
                                        minRotation: 0
                                    }}
                                }}
                            }},
                            plugins: {{
                                legend: {{
                                    display: true,
                                    labels: {{ font: {{ size: 12, family: 'Poppins' }} }}
                                }},
                                tooltip: {{
                                    enabled: true,
                                    bodyFont: {{ size: 12, family: 'Poppins' }},
                                    callbacks: {{
                                        label: function(context) {{
                                            return context.dataset.label + ': ' + formatNumber(context.raw);
                                        }}
                                    }}
                                }},
                                datalabels: {{
                                    anchor: 'end',
                                    align: 'top',
                                    font: {{ size: 12, family: 'Poppins', weight: '600' }},
                                    formatter: function(value) {{
                                        return formatNumber(value);
                                    }},
                                    color: '#1f2937'
                                }}
                            }},
                            layout: {{
                                padding: {{ bottom: 5, top: 30 }}
                            }}
                        }}
                    }});
                }} catch (e) {{
                    showError(`Error saat membuat chart: ${{e.toString()}}`);
                }}
            }}

            function renderReport() {{
                try {{
                    if (!Array.isArray(clusterStats)) {{
                        throw new Error('clusterStats is not an array');
                    }}
                    const root = document.getElementById('root');
                    root.innerHTML = `
                        <h1 class="text-3xl font-semibold mb-4 text-gray-800 section-title">Laporan Analisis Pelacakan Kendaraan</h1>
                        <p class="mb-4 text-gray-600">
                            Laporan ini menganalisis data pelacakan kendaraan menggunakan algoritma clustering K-Means untuk mengidentifikasi pola dalam perjalanan dan parkir. Laporan mencakup distribusi tipe peristiwa, lokasi teratas, distribusi cluster, dan analisis mendalam tentang pola yang ditemukan.
                        </p>

                        <div class="mt-6">
                            <h2 class="text-xl font-semibold mb-2 text-gray-700">Statistik Dasar</h2>
                            <p class="text-gray-600">
                                <strong>Jumlah Total Peristiwa:</strong> {stats['total_rows']}<br>
                                <strong>Rata-rata Durasi Peristiwa:</strong> {stats['avg_duration']} detik<br>
                                <strong>Rata-rata Jarak Tempuh:</strong> {stats['avg_mileage']} km
                            </p>
                        </div>

                        <div id="type-chart" class="chart-container">
                            <h2 class="text-xl font-semibold mb-2 text-gray-700">Distribusi Tipe Peristiwa</h2>
                            <canvas id="type-chart-canvas" class="chart"></canvas>
                        </div>

                        <div id="location-chart" class="chart-container">
                            <h2 class="text-xl font-semibold mb-2 text-gray-700">3 Lokasi Teratas</h2>
                            <canvas id="location-chart-canvas" class="chart"></canvas>
                        </div>

                        <div id="cluster-chart" class="chart-container">
                            <h2 class="text-xl font-semibold mb-2 text-gray-700">Distribusi Cluster Teratas</h2>
                            <canvas id="cluster-chart-canvas" class="chart"></canvas>
                        </div>

                        <div class="mt-6">
                            <h2 class="text-xl font-semibold mb-2 text-gray-700">Fakta Menarik</h2>
                            <p class="text-gray-600">
                                Lokasi <strong class="text-blue-600">{clean_string(location_chart_data[0]['location'] if location_chart_data else 'Tidak ada data')}</strong> adalah lokasi yang paling sering dikunjungi, dengan <strong class="text-blue-600">{location_chart_data[0]['count'] if location_chart_data else 0}</strong> peristiwa, menunjukkan bahwa lokasi ini kemungkinan merupakan titik parkir utama atau pusat aktivitas.
                            </p>
                        </div>

                        <div class="mt-6">
                            <h2 class="text-xl font-semibold mb-2 text-gray-700">Detail Cluster</h2>
                            <div class="grid grid-cols-1 gap-4">
    """
    # Generate cluster details statically
    for stat in cluster_stats:
        html_content += f"""
                                <div class="p-4 bg-gray-50 rounded-lg shadow-sm">
                                    <h3 class="text-lg font-semibold text-gray-700">Cluster {stat['cluster']}</h3>
                                    <p class="text-gray-600">
                                        <strong>Jumlah Peristiwa:</strong> {stat['count']}<br>
                                        <strong>Rata-rata Durasi:</strong> {stat['avg_duration']} detik<br>
                                        <strong>Rata-rata Jarak Tempuh:</strong> {stat['avg_mileage']} km<br>
                                        <strong>Tipe Peristiwa Terbanyak:</strong> {stat['most_common_type']}<br>
                                        <strong>Lokasi Terbanyak:</strong> {stat['most_common_location']}
                                    </p>
                                </div>
        """
    html_content += f"""
                            </div>
                        </div>

                        <div class="mt-6">
                            <h2 class="text-xl font-semibold mb-2 text-gray-700">Kesimpulan</h2>
                            <p class="text-gray-600">
                                Clustering K-Means mengungkapkan pola yang signifikan dalam pergerakan kendaraan. Lokasi-lokasi utama dan tipe peristiwa telah diidentifikasi, memberikan wawasan tentang perilaku kendaraan. Cluster yang dihasilkan menunjukkan variasi dalam durasi perjalanan, jarak tempuh, dan preferensi lokasi.
                            </p>
                        </div>

                        <div class="mt-6">
                            <h2 class="text-xl font-semibold mb-2 text-gray-700">Rekomendasi</h2>
                            <p class="text-gray-600">
                                1. <strong>Optimasi Rute:</strong> Fokuskan pada lokasi dengan frekuensi tinggi seperti {clean_string(location_chart_data[0]['location'] if location_chart_data else 'Tidak ada data')} untuk mengoptimalkan rute dan mengurangi kemacetan.<br>
                                2. <strong>Analisis Lebih Lanjut:</strong> Gunakan data koordinat spasial untuk analisis yang lebih mendalam tentang pola pergerakan.<br>
                                3. <strong>Pemantauan Anomali:</strong> Perhatikan cluster dengan durasi atau jarak tempuh yang sangat tinggi untuk mendeteksi potensi anomali, seperti perjalanan yang tidak efisien.
                            </p>
                        </div>
                    `;

                    createBarChart('type-chart-canvas', typeChartData, 'type', '#6366f1');
                    createBarChart('location-chart-canvas', locationChartData, 'location', '#34d399');
                    createBarChart('cluster-chart-canvas', clusterChartData, 'cluster', '#fb923c');
                }} catch (e) {{
                    showError(`Error saat membuat laporan: ${{e.toString()}}`);
                }}
            }}

            document.addEventListener('DOMContentLoaded', renderReport);
        </script>
    </body>
    </html>
    """
    
    with open("vehicle_tracking_report kmeans.html", "w", encoding="utf-8") as f:
        f.write(html_content)
    print("Pembuatan laporan HTML selesai.")
    return html_content

# Main execution
def main():
    print("Starting vehicle tracking analysis...")
    
    # Langkah 1: Fetch data
    data = fetch_data_from_snowflake()
    if data is None:
        print("Failed to fetch data. Exiting.")
        return
    
    # Langkah 2: Preprocess data
    processed_data, le_type, le_location = preprocess_data(data)
    if processed_data is None:
        print("Failed to preprocess data. Exiting.")
        return
    
    # Langkah 3: Apply clustering
    clustered_data = apply_dbscan(processed_data)
    if clustered_data is None:
        print("Failed to apply clustering. Exiting.")
        return
    
    # Langkah 4: Generate report
    generate_html_report(clustered_data, le_type, le_location)
    print("Vehicle tracking analysis completed.")

if __name__ == "__main__":
    main()