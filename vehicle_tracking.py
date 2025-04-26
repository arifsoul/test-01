import snowflake.connector
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler, LabelEncoder
import datetime
import pandas as pd
import re
import html
import json

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
    try:
        cursor = conn.cursor()
        query = "SELECT * FROM tr_track"
        cursor.execute(query)
        columns = [col[0] for col in cursor.description]
        results = cursor.fetchall()
        df = pd.DataFrame(results, columns=columns)
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
    filtered_data = data[
        data["UNIT"].notnull() & 
        data["TYPE"].notnull() & 
        data["DATE_TIME_BEGIN"].notnull() & 
        data["DATE_TIME_END"].notnull()
    ]
    
    if filtered_data.empty:
        print("No valid data after filtering.")
        return None, None, None
    
    processed_data = []
    for _, row in filtered_data.iterrows():
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
    for row in processed_data:
        row["TYPE_ENCODED"] = le_type.fit_transform([row["TYPE"]])[0]
        row["INITIAL_LOCATION_ENCODED"] = le_location.fit_transform([row["INITIAL_LOCATION"]])[0]
    
    return processed_data, le_type, le_location

# Apply DBSCAN clustering
def apply_dbscan(data):
    if not data:
        return None
    features = [
        [row["DURATION_SECONDS"], row["MILEAGE"], row["HOUR_OF_DAY"], row["DAY_OF_WEEK"],
         row["TYPE_ENCODED"], row["INITIAL_LOCATION_ENCODED"]]
        for row in data
    ]
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    clusters = dbscan.fit_predict(scaled_features)
    for i, row in enumerate(data):
        row["CLUSTER"] = clusters[i]
    return data

# Generate HTML report with Chart.js
def generate_html_report(data, le_type, le_location):
    if not data:
        print("No data to generate report.")
        return None
    
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
    for row in data:
        type_counts[row["TYPE"]] = type_counts.get(row["TYPE"], 0) + 1
        location_counts[row["INITIAL_LOCATION"]] = location_counts.get(row["INITIAL_LOCATION"], 0) + 1
        cluster_counts[row["CLUSTER"]] = cluster_counts.get(row["CLUSTER"], 0) + 1
    
    # Batasi cluster ke top 3 dan kelompokkan sisanya ke "Others"
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
        {"cluster": clean_string("Others" if k == -1 else str(k)), "count": v} for k, v in top_clusters
    ]
    
    try:
        type_chart_json = json.dumps(type_chart_data, ensure_ascii=False)
        location_chart_json = json.dumps(location_chart_data, ensure_ascii=False)
        cluster_chart_json = json.dumps(cluster_chart_data, ensure_ascii=False)
    except Exception as e:
        print(f"Error serializing JSON: {e}")
        return None
    
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Vehicle Tracking Analysis Report</title>
        <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
        <script src="https://cdn.tailwindcss.com"></script>
        <style>
            body {{ font-family: Arial, sans-serif; font-size: 14px; }}
            .chart-container {{ margin: 10px 0; max-width: 500px; }}
            .error {{ color: red; font-weight: bold; }}
            .chart {{ width: 100%; height: 200px; }}
        </style>
    </head>
    <body class="bg-gray-100 p-6">
        <div id="root" class="max-w-4xl mx-auto bg-white p-6 rounded-lg shadow-md"></div>
        <div id="error" class="error" style="display: none;"></div>
        <script>
            const typeChartData = {type_chart_json};
            const locationChartData = {location_chart_json};
            const clusterChartData = {cluster_chart_json};

            function showError(message) {{
                const errorDiv = document.getElementById('error');
                errorDiv.style.display = 'block';
                errorDiv.textContent = message;
            }}

            function createBarChart(canvasId, data, labelKey, color) {{
                try {{
                    const truncatedLabels = data.map(item => {{
                        const label = item[labelKey];
                        return label.length > 15 ? label.substring(0, 12) + '...' : label;
                    }});
                    const ctx = document.getElementById(canvasId).getContext('2d');
                    new Chart(ctx, {{
                        type: 'bar',
                        data: {{
                            labels: truncatedLabels,
                            datasets: [{{
                                label: 'Count',
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
                                y: {{
                                    beginAtZero: true,
                                    title: {{ display: true, text: 'Count' }}
                                }},
                                x: {{
                                    title: {{ display: true, text: labelKey.charAt(0).toUpperCase() + labelKey.slice(1) }},
                                    ticks: {{
                                        font: {{ size: 8 }},
                                        autoSkip: true,
                                        maxTicksLimit: 5,
                                        maxRotation: 30,
                                        minRotation: 0
                                    }}
                                }}
                            }},
                            plugins: {{
                                legend: {{ display: true }},
                                tooltip: {{ enabled: true }}
                            }},
                            layout: {{
                                padding: {{ bottom: 10 }}
                            }}
                        }}
                    }});
                }} catch (e) {{
                    showError(`Error rendering chart: ${{e.toString()}}`);
                }}
            }}

            function renderReport() {{
                try {{
                    const root = document.getElementById('root');
                    root.innerHTML = `
                        <h1 class="text-3xl font-bold mb-4">Vehicle Tracking Analysis Report</h1>
                        <p class="mb-4">
                            This report analyzes vehicle tracking data using DBSCAN clustering to identify patterns
                            in trips and parking events. Key insights include frequent locations, event types,
                            and cluster distributions.
                        </p>
                        <div id="type-chart" class="chart-container">
                            <h2 class="text-xl font-semibold mb-2">Event Type Distribution</h2>
                            <canvas id="type-chart-canvas" class="chart"></canvas>
                        </div>
                        <div id="location-chart" class="chart-container">
                            <h2 class="text-xl font-semibold mb-2">Top 3 Locations</h2>
                            <canvas id="location-chart-canvas" class="chart"></canvas>
                        </div>
                        <div id="cluster-chart" class="chart-container">
                            <h2 class="text-xl font-semibold mb-2">Top 3 Cluster Distribution</h2>
                            <canvas id="cluster-chart-canvas" class="chart"></canvas>
                        </div>
                        <div class="mt-6">
                            <h2 class="text-xl font-semibold mb-2">Interesting Fact</h2>
                            <p>
                                The location <strong>${clean_string(location_chart_data[0]['location'])}</strong> appears most frequently,
                                with <strong>${location_chart_data[0]['count']}</strong> events, indicating a potential parking hotspot.
                            </p>
                        </div>
                        <div class="mt-6">
                            <h2 class="text-xl font-semibold mb-2">Conclusion</h2>
                            <p>
                                The DBSCAN clustering reveals patterns in vehicle movements, with key locations and event types identified.
                                Outliers (noise points) may indicate anomalies, such as unusually long parking or high-mileage trips.
                                Future enhancements could include spatial coordinates for precise clustering.
                            </p>
                        </div>
                    `;

                    createBarChart('type-chart-canvas', typeChartData, 'type', '#8884d8');
                    createBarChart('location-chart-canvas', locationChartData, 'location', '#82ca9d');
                    createBarChart('cluster-chart-canvas', clusterChartData, 'cluster', '#ff7300');
                }} catch (e) {{
                    showError(`Error rendering application: ${{e.toString()}}`);
                }}
            }}

            document.addEventListener('DOMContentLoaded', renderReport);
        </script>
    </body>
    </html>
    """
    
    with open("vehicle_tracking_report.html", "w", encoding="utf-8") as f:
        f.write(html_content)
    print("Report generated: vehicle_tracking_report.html")
    return html_content

# Main execution
def main():
    data = fetch_data_from_snowflake()
    if data is None:
        print("Failed to fetch data. Exiting.")
        return
    processed_data, le_type, le_location = preprocess_data(data)
    if processed_data is None:
        print("Failed to preprocess data. Exiting.")
        return
    clustered_data = apply_dbscan(processed_data)
    if clustered_data is None:
        print("Failed to apply clustering. Exiting.")
        return
    generate_html_report(clustered_data, le_type, le_location)

if __name__ == "__main__":
    main()